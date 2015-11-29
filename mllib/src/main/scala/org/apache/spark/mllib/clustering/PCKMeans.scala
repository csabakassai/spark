package org.apache.spark.mllib.clustering

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.PCKMeans.ClusterCenter
import org.apache.spark.mllib.linalg.{BLAS, Vector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkContext}

import scala.collection.mutable.{Map => MutableMap}

/**
*  Created by cskassai on 12/10/15.
*/
class PCKMeans extends KMeans {

  type ClusterIndex = Int

  type ComponentId = Long

  def setPunishmentFactor(punishmentFactor: Double): PCKMeans = {
    this.punishmentFactor = punishmentFactor
    this
  }


  var punishmentFactor : Double = 2


  def run(data: RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): KMeansModel = {



    val (filteredAndNormalizedData, vectorComponentIdMap, elementsByComponents, cannotLinkComponentMap, centers) = PCKMeansInitializator.init(k, runs, data, mustLinkConstraints, cannotLinkConstraints)

    filteredAndNormalizedData.persist()

    logInfo(s"""Initial centers: ${centers.mkString("; ")}""")

    val model = runAlgorithm(filteredAndNormalizedData, vectorComponentIdMap, elementsByComponents, cannotLinkComponentMap, centers)

    filteredAndNormalizedData.unpersist()

    model

  }

  private def runAlgorithm(filteredAndNormalizedData: RDD[VectorWithNorm],
                           vectorComponentIdMap: Map[VectorWithNorm, Long],
                           elementsByComponents:  Map[ComponentId, Seq[VectorWithNorm]],
                           cannotLinkComponentMap: Map[Long, Set[Long]],
                           initialCenters: Map[ClusterIndex, ClusterCenter]): KMeansModel = {

    val sc: SparkContext = filteredAndNormalizedData.context

    var iteration = 0
    var currentCenters = initialCenters
    var centersChanged = true

    val constrainedElementClusters: MutableMap[VectorWithNorm, ClusterIndex] = MutableMap.empty
    
    while ( iteration < maxIterations && centersChanged ) {

      logDebug(s"Starting iteration $iteration ...")

      val broadcastedCenters = sc.broadcast(currentCenters)
      val costAccumulator = sc.accumulator(0.0)

      val normalSumAndCountByCenterPerPartition: RDD[(ClusterIndex, (Vector, Long))] = filteredAndNormalizedData.mapPartitions ( PCKMeans.calculatePartitionCentersAndContribution(_, broadcastedCenters, costAccumulator) )
      logTrace(s"Normal sum and count per partition: ${normalSumAndCountByCenterPerPartition.collect().mkString("; ")}")

      val constrainedSumAndCountByCenter : Seq[(ClusterIndex, (Vector, Long))] = PCKMeans.calculateConstrainedCentersAndContribution(currentCenters, costAccumulator, vectorComponentIdMap,elementsByComponents, cannotLinkComponentMap,constrainedElementClusters, punishmentFactor )
      logTrace(s"Constained sum and count per partition: ${constrainedSumAndCountByCenter.mkString("; ")}")

      val sumAndCountByCenterPerPartition = normalSumAndCountByCenterPerPartition ++ sc.parallelize(constrainedSumAndCountByCenter)

      val contributionsByCenter: Map[ClusterIndex, (Vector, Long)] = sumAndCountByCenterPerPartition.reduceByKey((elem1, elem2) => {
        (KMeansUtil.addVectors(elem1._1, elem2._1), elem1._2 + elem2._2)
      }).collectAsMap().toMap

      val newCenters: Map[ClusterIndex, ClusterCenter] = contributionsByCenter.mapValues(contribution => {

        val newCenter: Vector = contribution._1.copy

        BLAS.scal(1.0 / contribution._2, newCenter)
        new VectorWithNorm(newCenter)
      })

      logInfo(s"Old centers: ${currentCenters.mkString("; ")}")
      logInfo(s"""Centers after $iteration. iteration: ${newCenters.mkString("; ")}""")
      centersChanged = newCenters.map(newCenterWithIndex => {
        val oldCenter = currentCenters.get(newCenterWithIndex._1).get
        val newCenter = newCenterWithIndex._2
        KMeansUtil.fastSquaredDistance(newCenter, oldCenter) > epsilon * epsilon

      }).toSeq.contains(true)

      currentCenters = newCenters.toSeq.toMap
      iteration += 1
      
    }

    logInfo(s"Clustering finished after $iteration iteration")

    new KMeansModel(currentCenters.values.map(_.vector).toArray)
  }

}

class PCKMeansContext {

  val centersByIndex = MutableMap.empty

}

object PCKMeans {

  type ClusterCenter = VectorWithNorm

  type ComponentId = Long

  type ClusterIndex = Int

  def train( data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             mustLinkConstraints: Set[Constraint],
             cannotLinkConstraints: Set[Constraint],
             punishmentFactor: Double = 2.0 ): KMeansModel = {
    new PCKMeans().setK(k).setPunishmentFactor(punishmentFactor)
      .setMaxIterations(maxIterations)
      .setRuns(1)
      .setInitializationMode(null)
      .run(data, mustLinkConstraints, cannotLinkConstraints)
  }

  def calculatePartitionCentersAndContribution( elements: Iterator[VectorWithNorm],
                                                broadcastedCenters: Broadcast[Map[ClusterIndex, ClusterCenter]],
                                                costAccumulator: Accumulator[Double] ): Iterator[( ClusterIndex, (Vector, Long))] = {

    val currentCenters = broadcastedCenters.value

    val elementsWithClosestCenter = elements.map(element => {
                                                              val (closestIndex, _, cost) = findClosest(currentCenters, element)
                                                              costAccumulator.add(cost)
                                                              (element, closestIndex)}
    ).toSeq

    val elementByClosesCenter: Map[ClusterIndex, Seq[(VectorWithNorm, ClusterIndex)]] = elementsWithClosestCenter.groupBy( _._2 )

    val elementsAndCostsByCenterIndex: Map[ClusterIndex, Seq[Vector]] = elementByClosesCenter.mapValues( elementWithClosestCenterAndCost => elementWithClosestCenterAndCost.map( _._1.vector ))

    val sumAndCountByCenterIndex: Map[ClusterIndex, (Vector, Long)] = elementsAndCostsByCenterIndex.mapValues { elementsWithCosts =>

      val vectorSum: Vector = elementsWithCosts.reduce(KMeansUtil.addVectors)
      (vectorSum, elementsWithCosts.size.toLong)
    }
    sumAndCountByCenterIndex.toSeq.iterator

  }

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   */
  def findClosest(centers: Map[ClusterIndex, ClusterCenter],
                                  point: VectorWithNorm): (Int, VectorWithNorm, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestCenter: VectorWithNorm = null
    var bestIndex = 0
    for( (index, center) <- centers ) {
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = KMeansUtil.fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestCenter = center
          bestIndex = index
        }
      }
    }
    (bestIndex, bestCenter, bestDistance)
  }

  def calculateConstrainedCentersAndContribution(currentCenters: Map[ClusterIndex, ClusterCenter],
                                                 costAccumulator: Accumulator[Double],
                                                 componentByVector:  Map[VectorWithNorm, ComponentId],
                                                 elementsByComponents: Map[ComponentId, Seq[VectorWithNorm]],
                                                 cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                                                 constrainedElementClusters: MutableMap[VectorWithNorm, ClusterIndex],
                                                 punishmentFactor : Double): Seq[( ClusterIndex, (Vector, Long))] = {


    
    
    val elementsWithClosestCenter: Seq[(VectorWithNorm, ClusterIndex)] = componentByVector.map( element => {
      val (cluster, cost) = findClosestConstrained(currentCenters, element._1, componentByVector, elementsByComponents, cannotLinkComponentMap, constrainedElementClusters, punishmentFactor)
      costAccumulator.add(cost)
      (element._1, cluster)
    }).toSeq

    val elementByClosesCenter: Map[ClusterIndex, Seq[(VectorWithNorm, ClusterIndex)]] = elementsWithClosestCenter.groupBy( _._2 )

    val elementsAndCostsByCenterIndex: Map[ClusterIndex, Seq[Vector]] = elementByClosesCenter.mapValues( elementWithClosestCenterAndCost => elementWithClosestCenterAndCost.map( _._1.vector) )

    val sumAndCountByCenterIndex: Map[ClusterIndex, (Vector, Long)] = elementsAndCostsByCenterIndex.mapValues { elementsWithCosts =>

      val vectorSum: Vector = elementsWithCosts.reduce(KMeansUtil.addVectors)
      (vectorSum, elementsWithCosts.size.toLong)
    }

    
    sumAndCountByCenterIndex.toSeq

  }

  def findClosestConstrained(centers: Map[ClusterIndex, ClusterCenter],
                             point: VectorWithNorm,
                             componentByVector:  Map[VectorWithNorm, ComponentId],
                             elementsByComponents: Map[ComponentId, Seq[VectorWithNorm]],
                             cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                             currentConstrainedClusters: MutableMap[VectorWithNorm, ClusterIndex],
                             punishmentFactor : Double): ( ClusterIndex, Double ) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex: Int = -1
    var bestCenter: ClusterCenter = null
    val currentElementsByClusterCenter: Map[ClusterIndex, Seq[VectorWithNorm]] = currentConstrainedClusters.groupBy(_._2).mapValues( _.keys.toSeq)

    for( (index, center) <- centers ) {
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = KMeansUtil.fastSquaredDistance(center, point)

        val elementsInTheSameCluster = currentElementsByClusterCenter.getOrElse(index, Seq.empty)

        val currentComponentId = componentByVector.get(point).get

        val elementsInTheSameComponent = elementsByComponents.get(currentComponentId).get
        val cannotLinkViolations = calculateCannotLinkPunishmentFactor( point, componentByVector, cannotLinkComponentMap, elementsInTheSameCluster )
        val mustLinkViolations = calculateMustLinkPunishmentFactor(index, point, componentByVector, currentConstrainedClusters, elementsInTheSameComponent )


        import scala.math.pow

        val violatingElemNumber = cannotLinkViolations + mustLinkViolations
        val punishment: Double = pow(punishmentFactor, violatingElemNumber)
        val calculatedDistance = distance * punishment * punishment

        if (calculatedDistance < bestDistance) {
          bestDistance = calculatedDistance
          bestCenter = center
          bestIndex = index
        }
      }
    }

    currentConstrainedClusters += (point -> bestIndex)
    
    (bestIndex, bestDistance)
  }

  def calculateCannotLinkPunishmentFactor( element: VectorWithNorm,
                                           componentByVector:  Map[VectorWithNorm, ComponentId],
                                           cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                                           elementsInTheSameCluster: Seq[VectorWithNorm] ) : Int = {

    val elementComponentId = componentByVector.get(element).get

    val cannotLinkComponentIdOption: Option[Set[ComponentId]] = cannotLinkComponentMap.get(elementComponentId)
    val hasCannotLinkConstrain = cannotLinkComponentIdOption.isDefined

    var violatingElemNumber = 0

    if ( hasCannotLinkConstrain ) {

      val cannotLinkComponentIds: Set[ComponentId] = cannotLinkComponentIdOption.get

      elementsInTheSameCluster.foreach( elementInTheSameCluster => {
        val elementInTheSameClusterComponentId = componentByVector.get(elementInTheSameCluster).get

        if(cannotLinkComponentIds.contains(elementInTheSameClusterComponentId)) {
          violatingElemNumber += 1
        }

      })
    }
    
    violatingElemNumber

  }

  def calculateMustLinkPunishmentFactor( cluster: ClusterIndex,
                                         element: VectorWithNorm,
                                         componentByVector:  Map[VectorWithNorm, ComponentId],
                                         currentContrainedClusters: MutableMap[VectorWithNorm, ClusterIndex],
                                         elementsInTheSameComponent: Seq[VectorWithNorm] ) : Int = {
    
    val hasMustLinkConstraint = elementsInTheSameComponent.size > 1

    var violatingElemNumber = 0

    if ( hasMustLinkConstraint ) {
      for(elementInTheSameComponent <- elementsInTheSameComponent
          if elementInTheSameComponent != element) {
        val elementInTheSameComponentClusterIndex = currentContrainedClusters.get(elementInTheSameComponent)

        violatingElemNumber += elementInTheSameComponentClusterIndex.count(_ != cluster)
      }
    }
    
    violatingElemNumber
  }

}

object KMeansUtil {
  /**
   * Returns the squared Euclidean distance between two vectors computed by
   * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
   */
  private[clustering] def fastSquaredDistance(
                                               v1: VectorWithNorm,
                                               v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  def addVectors( v1: Vector, v2: Vector): Vector = {
    val vectorToAdd2: Vector = v2.copy
    BLAS.axpy(1.0, v1, vectorToAdd2)
    vectorToAdd2
  }
}


case class Constraint ( elements: ( Vector, Vector ), weight: Double = 1 ) extends Serializable



