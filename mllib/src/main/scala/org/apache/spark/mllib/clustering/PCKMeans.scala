package org.apache.spark.mllib.clustering


import com.google.common.base.Stopwatch
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.PCKMeans.ClusterCenter
import org.apache.spark.mllib.clustering.PCKMeansInitializator._
import org.apache.spark.mllib.linalg.{Vectors, BLAS, Vector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkContext}
import org.github.jamm.MemoryMeter
import org.slf4j.{MarkerFactory, Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.mutable.{Map => MutableMap}

/**
*  Created by cskassai on 12/10/15.
*/
class PCKMeans extends KMeans {

  type ClusterIndex = Int

  type ComponentId = Long

  private val mlog: Logger = LoggerFactory.getLogger("pckmeans.memory")
  private val TIME_LOG: Logger = LoggerFactory.getLogger("pckmeans.time")

  def setPunishmentFactor(punishmentFactor: Double): PCKMeans = {
    this.punishmentFactor = punishmentFactor
    this
  }


  var punishmentFactor : Double = 2

  var mustLinkSize = 0l
  var cannotLinkSize = 0l
  var mustlinkComponentSize = 0l

  var tempmem = 0l


  def run(data: RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): KMeansModel = {

    TIME_LOG.info("==================================================")
    mustLinkSize = mustLinkConstraints.size
    cannotLinkSize = cannotLinkConstraints.size

    val watch: Stopwatch = new Stopwatch()
    watch.start()
    val (filteredAndNormalizedData, vectorComponentIdMap, elementsByComponents, cannotLinkComponentMap, centers) = PCKMeansInitializator.init(k, runs, data, mustLinkConstraints, cannotLinkConstraints)


    mustlinkComponentSize = elementsByComponents.keys.size
    filteredAndNormalizedData.persist()

    logInfo(s"""Initial centers: ${centers.mkString("; ")}""")
    TIME_LOG.info(s"Initialization: ${watch.stop()}")
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
    val alma: MutableMap[ClusterIndex, Set[VectorWithNorm]] = MutableMap.empty
    
    while ( iteration < maxIterations && centersChanged ) {

      logDebug(s"Starting iteration $iteration ...")

      val broadcastedCenters = sc.broadcast(currentCenters)
      val costAccumulator = sc.accumulator(0.0)

      val watch: Stopwatch = new Stopwatch()

      watch.start();
      val normalSumAndCountByCenterPerPartition: RDD[(ClusterIndex, (Vector, Long))] = filteredAndNormalizedData.mapPartitions ( PCKMeans.calculatePartitionCentersAndContribution(_, broadcastedCenters, costAccumulator) )
      logTrace(s"Normal sum and count per partition: ${normalSumAndCountByCenterPerPartition.collect().mkString("; ")}")
      TIME_LOG.info(s"NORMAL: ${watch}")

      watch.reset().start()
      val (constrainedSumAndCountByCenter, actualTemp) : (Seq[(ClusterIndex, (Vector, Long))], Long) = PCKMeans.calculateConstrainedCentersAndContribution(currentCenters, costAccumulator, vectorComponentIdMap,elementsByComponents, cannotLinkComponentMap,constrainedElementClusters, alma, punishmentFactor )
      logTrace(s"Constained sum and count per partition: ${constrainedSumAndCountByCenter.mkString("; ")}")
      TIME_LOG.info(s"CONSTRAINED: ${watch}")

//      if(actualTemp > tempmem) {
//        tempmem = actualTemp
//      }
      watch.reset().start()
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
      TIME_LOG.info(s"SUM: ${watch}")
      
    }

    logWarning(s"Clustering finished after $iteration iteration")
    
    

//    val meter = new MemoryMeter();
//    val sumData = (vectorComponentIdMap, elementsByComponents, cannotLinkComponentMap, constrainedElementClusters)
//    val deep: Long = meter.measureDeep(sumData)


    if(centersChanged) {
      logWarning("The clustering stopped because of the maxiteration constraint!")
    }
//    mlog.warn(s"$mustLinkSize;$cannotLinkSize;$mustlinkComponentSize;${vectorComponentIdMap.keys.size};$tempmem;${meter.measureDeep(vectorComponentIdMap)};${meter.measureDeep(elementsByComponents)};${meter.measureDeep(cannotLinkComponentMap)};${meter.measureDeep(constrainedElementClusters)};$deep")

    new KMeansModel(currentCenters.values.map(_.vector).toArray)
  }

}

class PCKMeansContext {

  val centersByIndex = MutableMap.empty

}

object PCKMeans {

  private val TIME_LOG: Logger = LoggerFactory.getLogger("pckmeans.time")
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

    val dims: Int = currentCenters.values.head.vector.size
    val k = currentCenters.keys.size
    val sums: Array[Vector] = Array.fill(k)(Vectors.zeros(dims))
    val counts: Array[Long] = Array.fill(k)(0l)

    elements.foreach( elem => {
      val (closestIndex, _, cost) = findClosest(currentCenters, elem)
      costAccumulator += cost
      val currentSum = sums(closestIndex)
      BLAS.axpy(1.0, elem.vector, currentSum)
      counts(closestIndex) = counts(closestIndex) + 1

    })
    currentCenters.map(entry => (entry._1, (sums(entry._1), counts(entry._1)))).iterator

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
                                                 alma: MutableMap[ClusterIndex, Set[VectorWithNorm]],
                                                 punishmentFactor : Double): (Seq[( ClusterIndex, (Vector, Long))], Long) = {



    val watch: Stopwatch = new Stopwatch()
    watch.start()
    val elementsWithClosestCenter: Seq[(VectorWithNorm, ClusterIndex)] = componentByVector.map( element => {
      val (cluster, cost) = findClosestConstrained(currentCenters, element._1, componentByVector, elementsByComponents, cannotLinkComponentMap, constrainedElementClusters, alma, punishmentFactor)
      costAccumulator.add(cost)
      (element._1, cluster)
    }).toSeq

    TIME_LOG.info(s"closest: ${watch}")

    watch.reset().start();;

    val elementByClosesCenter: Map[ClusterIndex, Seq[(VectorWithNorm, ClusterIndex)]] = elementsWithClosestCenter.groupBy( _._2 )

    val elementsAndCostsByCenterIndex: Map[ClusterIndex, Seq[Vector]] = elementByClosesCenter.mapValues( elementWithClosestCenterAndCost => elementWithClosestCenterAndCost.map( _._1.vector) )

    val sumAndCountByCenterIndex: Map[ClusterIndex, (Vector, Long)] = elementsAndCostsByCenterIndex.mapValues { elementsWithCosts =>

      val vectorSum: Vector = elementsWithCosts.reduce(KMeansUtil.addVectors)
      (vectorSum, elementsWithCosts.size.toLong)
    }

    val toSeq: Seq[(ClusterIndex, (Vector, ComponentId))] = sumAndCountByCenterIndex.toSeq

    TIME_LOG.info(s"else: ${watch}")


//    val memoryMeter = new MemoryMeter()
//    val actualTemp: ComponentId = memoryMeter.measureDeep((toSeq, sumAndCountByCenterIndex, elementsAndCostsByCenterIndex, elementByClosesCenter, elementByClosesCenter))

    (toSeq, 0)

  }

  def findClosestConstrained(centers: Map[ClusterIndex, ClusterCenter],
                             point: VectorWithNorm,
                             componentByVector:  Map[VectorWithNorm, ComponentId],
                             elementsByComponents: Map[ComponentId, Seq[VectorWithNorm]],
                             cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                             currentConstrainedClusters: MutableMap[VectorWithNorm, ClusterIndex],
                             currentElementsByClusterCenter: MutableMap[ClusterIndex, Set[VectorWithNorm]],
                             punishmentFactor : Double): ( ClusterIndex, Double ) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex: Int = -1
    var bestCenter: ClusterCenter = null

    for( (index, center) <- centers ) {
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = KMeansUtil.fastSquaredDistance(center, point)

        val emptySet: Set[VectorWithNorm] = Set.empty

        val elementsInTheSameCluster = currentElementsByClusterCenter.getOrElse(index, emptySet)
        val currentComponentId = componentByVector.get(point).get

        val elementsInTheSameComponent = elementsByComponents.get(currentComponentId).get
        val cannotLinkViolations = calculateCannotLinkPunishmentFactor( point, componentByVector, cannotLinkComponentMap, elementsInTheSameCluster )
        val mustLinkViolations = calculateMustLinkPunishmentFactor(index, point, componentByVector, currentConstrainedClusters, elementsInTheSameComponent )

        val violatingElemNumber = cannotLinkViolations + mustLinkViolations
        val calculatedDistance = distance + violatingElemNumber * punishmentFactor

        if (calculatedDistance < bestDistance) {
          bestDistance = calculatedDistance
          bestCenter = center
          bestIndex = index
        }
      }
    }

    val oldCluster: ClusterIndex = currentConstrainedClusters.getOrElse(point, -1)

    var e: Set[VectorWithNorm] = currentElementsByClusterCenter.getOrElse(oldCluster, Set.empty)
    e -= point
    if(oldCluster > 0) {
      currentElementsByClusterCenter += (oldCluster -> e)
    }

    var newSet: Set[VectorWithNorm] = currentElementsByClusterCenter.getOrElse(bestIndex, Set.empty)
    newSet += (point)
    currentElementsByClusterCenter += (bestIndex -> newSet)
    currentConstrainedClusters += (point -> bestIndex)

    
    (bestIndex, bestDistance)
  }

  def calculateCannotLinkPunishmentFactor( element: VectorWithNorm,
                                           componentByVector:  Map[VectorWithNorm, ComponentId],
                                           cannotLinkComponentMap: Map[ComponentId, Set[ComponentId]],
                                           elementsInTheSameCluster: Set[VectorWithNorm] ) : Int = {

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



