package org.apache.spark.mllib.clustering

import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

import com.google.common.base.Stopwatch
import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.clustering.ClusteringValidator.{ClusterIndex, PartitionIndex}
import org.apache.spark.mllib.clustering.KMeans._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD

import scala.collection.TraversableOnce

/**
*  Created by cskassai on 19/10/15.
*/
class PCKMeansSuite extends SparkFunSuite with MLlibTestSparkContext {

  val resultFilePath = "/Users/cskassai/Egyetem/Diplomaterv/experiments/"

  val cluster1 = Seq(
    //Cluster 1
    Vectors.dense(0.0, 0.0),
    Vectors.dense(0.0, 2.0),
    Vectors.dense(2.0, 0.0),
    Vectors.dense(2.0, 2.0))

  val cluster2 = Seq(
    //Cluster 1
    Vectors.dense(7.0, 0.0),
    Vectors.dense(9.0, 0.0),
    Vectors.dense(7.0, 2.0),
    Vectors.dense(9.0, 2.0))

  val constrainedPoint = Seq (
    Vectors.dense(4.0, 1.0)
  )

  test("two clusters without constraints") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val rdd = sc.parallelize(points, 3)

    for (initMode <- Seq(RANDOM, K_MEANS_PARALLEL)) {
      // Two iterations are sufficient no matter where the initial centers are.
      val model = KMeans.train(rdd, k = 2, maxIterations = 8, runs = 1, initMode)

      log.error(s"""InitMode: $initMode, centers: ${model.clusterCenters.mkString(", ")}""" )
      assertClusters(Seq (cluster1 ++ constrainedPoint, cluster2), model)
    }
  }

  test("two clusters with cannotLink") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val data = sc.parallelize(points, 3)

    val model = PCKMeans.train(data, 2, 4,  Set.empty, Set(Constraint((constrainedPoint.head, cluster1.head), 2.0)))
    log.error(s"""centers: ${model.clusterCenters.mkString(", ")}""" )
    assertClusters(Seq (cluster1, cluster2 ++ constrainedPoint), model)

  }

  test("two clusters with mustLink") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val data = sc.parallelize(points, 3)

    // Two iterations are sufficient no matter where the initial centers are.
    val model = PCKMeans.train(data, 2, 4, Set(Constraint((constrainedPoint.head, cluster2.head), 2.0)), Set.empty)
    log.error(s"""centers: ${model.clusterCenters.mkString(", ")}""" )
    assertClusters(Seq (cluster1, cluster2 ++ constrainedPoint), model)

  }


  def assertClusters ( expectedClusters: Seq[Seq[Vector]], model: KMeansModel): Unit = {
    for( expectedCluster <- expectedClusters ) {
      val sum: Vector = expectedCluster.reduce(KMeansUtil.addVectors)
      BLAS.scal(1.0/expectedCluster.size, sum)
      assert( model.clusterCenters.contains(sum))
    }
  }

  def convertAnyToString(any: Any): Seq[String] = {
    any match {
      case v: Vector => v.toArray.map(_.toString).toSeq
      case _ => Seq(any.toString)
    }
  }

  def convertProductToIndexedStringSet(product: Product):IndexedSeq[String] = {
    val map: Iterator[Seq[String]] = product.productIterator.map(convertAnyToString)
    map.flatten.toIndexedSeq
  }

  def writeClusteringResultToFile(resultIterator: Iterator[Product], filePath: String): Unit = {

    val lines: String = resultIterator.map( elem => convertProductToIndexedStringSet(elem).mkString(";") ).mkString("\n")

    scala.tools.nsc.io.File(filePath).writeAll(lines)

  }

  def validateClustering(elementsRDD: RDD[Vector], elementsWithPartitions: RDD[(Vector, Int)], kMeansModel: KMeansModel): (RDD[(Vector, Int, Int)], Double, Double) = {

    val result: RDD[(Vector, Int, Int)] = elementsWithPartitions.zip(kMeansModel.predict(elementsRDD)).map( elem => (elem._1._1, elem._1._2, elem._2))

    val toValidateBaseLine: RDD[(PartitionIndex, ClusterIndex)] = result.map(elem => (elem._2, elem._3)).persist()
    val fMeasure: Double = ClusteringValidator.calculateFMeasure(toValidateBaseLine)._1
    val purity: Double = ClusteringValidator.calculatePurity(toValidateBaseLine)._1

    (result, fMeasure, purity)
  }

  def calculateKMeansBaseLine( elementsWithPartitions: RDD[(Vector, PartitionIndex)], elementsRDD : RDD[(Vector)]):
                                  (RDD[(Vector, PartitionIndex, ClusterIndex)],
                                  Double,
                                  Double,
                                  RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))],
                                  RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))],
                                  Long) = {

    val stopwatch: Stopwatch = new Stopwatch().start()
    val kmeansModel: KMeansModel = KMeans.train(elementsRDD, 2, 100)
    val time: Long = stopwatch.stop.elapsed(TimeUnit.MILLISECONDS)

    val (result, fMeasure, purity) = validateClustering(elementsRDD, elementsWithPartitions, kmeansModel)

    val cartesianRDD: RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))] = result.cartesian(result).cache()

    val mustlinkSource: RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))] = cartesianRDD.filter(elem => elem._1._2 == elem._2._2 && elem._1._3 != elem._2._3).cache()
    val cannotlinkSource: RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))] = cartesianRDD.filter(elem => elem._1._2 != elem._2._2 && elem._1._3 == elem._2._3).cache()

    ( result, fMeasure, purity, mustlinkSource, cannotlinkSource, time )
  }

  test("Small big test") {
    val elements: Seq[(Vector, Int)] = ClusterGenerator.generate(
      EllipsoidClusterDescriptor(Vectors.dense(0, 0), Vectors.dense(4, 4), 1000),
      EllipsoidClusterDescriptor(Vectors.dense(0, 8), Vectors.dense(1, 1), 100))

    runExperiment(  data = elements,
      punishmentFactors = Seq(2, 10, 100),
      mustLinkPercentages = 0 until(100, 30),
      cannotLinkPercentages = 0 until(100, 30),
      experimentNumber =  5,
      "small_big")


  }

  def generateConstraints(source: RDD[((Vector, PartitionIndex, ClusterIndex), (Vector, PartitionIndex, ClusterIndex))], elemNumber: Int, percentage: Double): Set[Constraint] = {

    val constaints: Set[Constraint] = source.takeSample(withReplacement = false, (elemNumber * percentage/100).toInt ).map( elem => Constraint((elem._1._1, elem._2._1), 1)).toSet
    constaints
  }

  def runExperiment( data: Seq[(Vector, Int)],
                     punishmentFactors: TraversableOnce[Double],
                     mustLinkPercentages: TraversableOnce[Int],
                     cannotLinkPercentages: TraversableOnce[Int],
                     experimentNumber: Int,
                     experimentName: String): Unit = {

    val elementsWithPartitions: RDD[(Vector, Int)] = sc.parallelize(data).persist()
    val elementsRDD: RDD[Vector] = elementsWithPartitions.keys.persist()

    val format: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd-hhmm")
    val experimentPath = resultFilePath + experimentName + "_" + LocalDateTime.now().format(format) + "/"

    new File(experimentPath).mkdir()

    val ( kMeansResult, fMeasure, purity, mustlinkSource, cannotlinkSource, time ) = calculateKMeansBaseLine(elementsWithPartitions, elementsRDD)

    writeClusteringResultToFile(kMeansResult.toLocalIterator, experimentPath + "baseline.csv")

    val base: (Double, Double, Double, Double, Double, Long) = (-1, -1, -1, fMeasure, purity, time)

    val resultFileName = "ellipsoid_result"
    val baseLine: String = convertProductToIndexedStringSet(base).mkString(";")
    logInfo(s"Base: $baseLine")
    scala.tools.nsc.io.File(experimentPath + resultFileName).writeAll(baseLine + "\n")

    val elemNumber: Int = elementsRDD.count().toInt

    for(  punishmentFactor <- punishmentFactors;
          mustlinkPercentage <- mustLinkPercentages;
          cannotlinkPercentage <- cannotLinkPercentages;
          experimentIndex <- 0 until experimentNumber) {


      logWarning(s"Parameters: punishment = $punishmentFactor, mustlink = $mustlinkPercentage, cannotlink = $cannotlinkPercentage ")

      val cannotLinks: Set[Constraint] = generateConstraints(cannotlinkSource, elemNumber, cannotlinkPercentage)
      val mustLinks: Set[Constraint] = generateConstraints(mustlinkSource, elemNumber, mustlinkPercentage)

      val stopwatch: Stopwatch = new Stopwatch().start()
      val pckMeansModel: KMeansModel = PCKMeans.train(elementsRDD, 2, 100, mustLinks, cannotLinks, punishmentFactor )
      val time: Long = stopwatch.stop.elapsed(TimeUnit.MILLISECONDS)

      val resultRDD: RDD[(Vector, PartitionIndex, ClusterIndex)] = elementsWithPartitions.map(elem => (elem._1, elem._2, pckMeansModel.predict(elem._1))).persist()
      writeClusteringResultToFile(resultRDD.toLocalIterator, s"${experimentPath}ellipsoid_${punishmentFactor}_${mustlinkPercentage}_${cannotlinkPercentage}_$experimentIndex.csv")

      val (result, fMeasure, purity) = validateClustering(elementsRDD, elementsWithPartitions, pckMeansModel)

      val iterationResult = (punishmentFactor, mustlinkPercentage, cannotlinkPercentage, fMeasure, purity, time, experimentIndex)
      val resultLine: String = convertProductToIndexedStringSet(iterationResult).mkString(";")
      logInfo(resultLine)
      scala.tools.nsc.io.File(experimentPath + resultFileName).appendAll(resultLine + "\n")




    }
  }

  test("Ellipsoid tuning") {
    val clusterElemNumber = 1000
    val elements: Seq[(Vector, Int)] = ClusterGenerator.generate(
      EllipsoidClusterDescriptor(Vectors.dense(0, 0), Vectors.dense(4, 16), clusterElemNumber),
      EllipsoidClusterDescriptor(Vectors.dense(10, 0), Vectors.dense(4, 16), clusterElemNumber))

    runExperiment(  data = elements,
                    punishmentFactors = Seq(2, 10, 100),
                    mustLinkPercentages = 0 until(100, 30),
                    cannotLinkPercentages = 0 until(100, 30),
                    experimentNumber =  2,
                    "ellipsoid")


  }


}
