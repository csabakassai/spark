package org.apache.spark.mllib.clustering

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.PCKMeans.ClusterCenter
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkContext}

import scala.collection
import scala.collection.mutable.{Map => MutableMap}

/**
 * Created by cskassai on 12/10/15.
 */
class PCKMeans extends KMeans {


  def run(data: RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): KMeansModel = {

    val (filteredAndNormalizedData, vectorComponentIdMap, cannotLinkComponentMap, centers) = PCKMeansInitializator.init(k, runs, data, mustLinkConstraints, cannotLinkConstraints)

    filteredAndNormalizedData.persist()

    val model = runAlgorithm(filteredAndNormalizedData, vectorComponentIdMap, cannotLinkComponentMap, centers)

    filteredAndNormalizedData.unpersist()

    model

  }

  private def runAlgorithm(filteredAndNormalizedData: RDD[VectorWithNorm], vectorComponentIdMap: Map[VectorWithNorm, Long], cannotLinkComponentMap: Map[Long, Set[Long]], initialCenters: Array[VectorWithNorm]): KMeansModel = {

    val sc: SparkContext = filteredAndNormalizedData.context

    var iteration = 0
    val currentCenters = initialCenters
    var centersChanged = true

    val constrainedElementClusters: Map[VectorWithNorm, ClusterCenter] = Map.empty
    
    while ( iteration < maxIterations && centersChanged ) {
      
      val centerToCenterIndexMap: Map[ClusterCenter, Int] = currentCenters.zip(Stream.from(0)).groupBy(_._1).mapValues(_(0)._2)

      val elementsByClusterCenter: Map[ClusterCenter, Seq[VectorWithNorm]] = Map.empty
      val broadcastedCenters = sc.broadcast(currentCenters)
      val costAccumulator = sc.accumulator(0.0)

      val sumAndCountByCenterPerPartition: RDD[(Int, (Vector, Long))] = filteredAndNormalizedData.mapPartitions ( PCKMeans.calculatePartitionCentersAndContribution(_, broadcastedCenters, costAccumulator, KMeans.findClosest) )

      val constrainedSumAndCountByCenter : Seq[(ClusterCenter, (Vector, Long))] = PCKMeans.calculateConstrainedCentersAndContribution(currentCenters, costAccumulator, vectorComponentIdMap, cannotLinkComponentMap,constrainedElementClusters, elementsByClusterCenter )

      sumAndCountByCenterPerPartition ++ sc.parallelize(constrainedSumAndCountByCenter.map( elem => {
        val clusterCenterIndex = centerToCenterIndexMap.get(elem._1).get
        (clusterCenterIndex, elem._2)
      }))

      val contributionsByCenter: collection.Map[Int, (Vector, Long)] = sumAndCountByCenterPerPartition.reduceByKey((elem1, elem2) => {
        (KMeansUtil.addVectors(elem1._1, elem2._1), elem1._2 + elem2._2)
      }).collectAsMap()

      val newCenters: collection.Map[Int, VectorWithNorm] = contributionsByCenter.mapValues(conntribution => {

        val newCenter: Vector = conntribution._1.copy

        BLAS.scal(1.0 / conntribution._2, newCenter)
        new VectorWithNorm(newCenter)
      })

      centersChanged = newCenters.map(newCenterWithIndex => {
        val clusterIndex = newCenterWithIndex._1
        val oldCenter = currentCenters(clusterIndex)
        val newCenter = newCenterWithIndex._2
        currentCenters(clusterIndex) = newCenter
        KMeansUtil.fastSquaredDistance(newCenter, oldCenter) > epsilon * epsilon

      }).toSeq.contains(true)

      iteration += 1
      
    }

    new KMeansModel(currentCenters.map(_.vector))
  }

}

object PCKMeans {


  type ClusterCenter = VectorWithNorm

  type ComponentId = Long

  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             mustLinkConstraints: Set[Constraint],
             cannotLinkConstraints: Set[Constraint]): KMeansModel = {
    new PCKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setRuns(1)
      .setInitializationMode(null)
      .run(data, mustLinkConstraints, cannotLinkConstraints)
  }

  def calculatePartitionCentersAndContribution( elements: Iterator[VectorWithNorm],
                                                broadcastedCenters: Broadcast[Array[VectorWithNorm]],
                                                costAccumulator: Accumulator[Double],
                                                findClosestFunction: (TraversableOnce[VectorWithNorm], VectorWithNorm) => (Int, Double) ): Iterator[( Int, (Vector, Long))] = {

    val currentCenters = broadcastedCenters.value

    val elementsWithClosestCenter = elements.map(element => {
                                                              val (closestCenter, cost) = findClosestFunction(currentCenters, element)
                                                              costAccumulator.add(cost)
                                                              (element, closestCenter)}
    ).toSeq

    val elementByClosesCenter: Map[Int, Seq[(VectorWithNorm, Int)]] = elementsWithClosestCenter.groupBy( _._2 )

    val elementsAndCostsByCenterIndex: Map[Int, Seq[Vector]] = elementByClosesCenter.mapValues( elementWithClosestCenterAndCost => elementWithClosestCenterAndCost.map( e => (e._1.vector)) )

    val sumAndCountByCenterIndex: Map[Int, (Vector, Long)] = elementsAndCostsByCenterIndex.mapValues { elementsWithCosts =>

      val vectorSum: Vector = elementsWithCosts.reduce(KMeansUtil.addVectors)
      (vectorSum, elementsWithCosts.size.toLong)
    }
    sumAndCountByCenterIndex.toSeq.iterator

  }

  def calculateConstrainedCentersAndContribution(currentCenters: Array[VectorWithNorm],
                                                 costAccumulator: Accumulator[Double],
                                                 componentByVector:  Map[VectorWithNorm, ComponentId],
                                                 cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                                                 constrainedElementClusters: Map[VectorWithNorm, ClusterCenter],
                                                 elementsByClusterCenter: Map[ClusterCenter, Seq[VectorWithNorm]]): Seq[( ClusterCenter, (Vector, Long))] = {


    
    
    val elementsWithClosestCenter: Seq[(VectorWithNorm, ClusterCenter)] = componentByVector.map( element => {
      val (closestCenter, cost) = findClosestConstrained(currentCenters, element._1, componentByVector, cannotLinkComponentMap, constrainedElementClusters, elementsByClusterCenter)
      costAccumulator.add(cost)
      (element._1, closestCenter)
    }).toSeq

    val elementByClosesCenter: Map[ClusterCenter, Seq[(VectorWithNorm, ClusterCenter)]] = elementsWithClosestCenter.groupBy( _._2 )

    val elementsAndCostsByCenterIndex: Map[ClusterCenter, Seq[Vector]] = elementByClosesCenter.mapValues( elementWithClosestCenterAndCost => elementWithClosestCenterAndCost.map( e => (e._1.vector)) )

    val sumAndCountByCenterIndex: Map[ClusterCenter, (Vector, Long)] = elementsAndCostsByCenterIndex.mapValues { elementsWithCosts =>

      val vectorSum: Vector = elementsWithCosts.reduce(KMeansUtil.addVectors)
      (vectorSum, elementsWithCosts.size.toLong)
    }
    
    
    sumAndCountByCenterIndex.toSeq

  }

  def findClosestConstrained(centers: TraversableOnce[VectorWithNorm],
                             point: VectorWithNorm,
                             componentByVector:  Map[VectorWithNorm, ComponentId],
                             cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                             currentContrainedClusters: Map[VectorWithNorm, ClusterCenter],
                             elementsByClusterCenter: Map[ClusterCenter, Seq[VectorWithNorm]]): ( ClusterCenter, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestCenter: VectorWithNorm = null;
    
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = KMeansUtil.fastSquaredDistance(center, point)

        val elementsInTheSameCluster = elementsByClusterCenter.get(center).getOrElse(Seq.empty)
        val cannotLinkViolations = calculateCannotLinkPunishmentFactor( point, componentByVector, cannotLinkComponentMap, elementsInTheSameCluster )
        val mustLinkViolations: Double = 1.0


        import scala.math.pow

        val punishmentFactor = 1.1
        val violatingElemNumber = cannotLinkViolations + mustLinkViolations
        val punishment: Double = pow(punishmentFactor, violatingElemNumber)
        val calculatedDistance = distance * punishment

        if (calculatedDistance < bestDistance) {
          bestDistance = calculatedDistance
          bestCenter = center
        }
      }
    }

    currentContrainedClusters + (point -> bestCenter)
    
    (bestCenter, bestDistance)
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

  def calculateMustLinkPunishmentFactor( center: ClusterCenter,
                                         element: VectorWithNorm,
                                         componentByVector:  Map[VectorWithNorm, ComponentId],
                                         currentContrainedClusters: Map[VectorWithNorm, ClusterCenter],
                                         elementsInTheSameComponent: Seq[VectorWithNorm] ) : Int = {
    
    val hasMustLinkConstraint = elementsInTheSameComponent.size > 1

    var violatingElemNumber = 0

    if ( hasMustLinkConstraint ) {
      elementsInTheSameComponent.foreach( elementInTheSameComponent => {
        val elementInTheSameComponentClusterCenter = currentContrainedClusters.get(elementInTheSameComponent).getOrElse(null)

        if(center != elementInTheSameComponentClusterCenter) {
          violatingElemNumber += 1
        }
      })
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



