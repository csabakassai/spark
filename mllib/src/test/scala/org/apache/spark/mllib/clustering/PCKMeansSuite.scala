package org.apache.spark.mllib.clustering

import java.io.{FileInputStream, File}
import java.text.SimpleDateFormat
import java.util
import java.util.Date

import java.util.concurrent.TimeUnit

import com.google.common.base.Stopwatch
import org.apache.commons.io.IOUtils
import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.clustering.ClusteringValidator.{ClusterIndex, PartitionIndex}
import org.apache.spark.mllib.clustering.KMeans._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.{Map => MutableMap}
import scala.collection.{Map, mutable, TraversableOnce}
import scala.io.Source

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

    val model = PCKMeans.train(data, 2, 4,  Set.empty, Set(Constraint((constrainedPoint.head, cluster1.head), 5.0)))
    log.error(s"""centers: ${model.clusterCenters.mkString(", ")}""" )
    assertClusters(Seq (cluster1, cluster2 ++ constrainedPoint), model)

  }

  test("two clusters with mustLink") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val data = sc.parallelize(points, 3)

    // Two iterations are sufficient no matter where the initial centers are.
    val model = PCKMeans.train(data, 2, 4, Set(Constraint((constrainedPoint.head, cluster2.head), 5.0)), Set.empty)
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

  def calculateKMeansBaseLine(k: Int, elementsWithPartitions: RDD[(Vector, PartitionIndex)], elementsRDD : RDD[(Vector)]):
                                  (RDD[(Vector, PartitionIndex, ClusterIndex)],
                                  Double,
                                  Double,
                                  Long) = {


    val stopwatch: Stopwatch = new Stopwatch().start()
    val kmeansModel: KMeansModel = KMeans.train(elementsRDD, k, 10000)
    val time: Long = stopwatch.stop.elapsed(TimeUnit.MILLISECONDS)

    val (result, fMeasure, purity) = validateClustering(elementsRDD, elementsWithPartitions, kmeansModel)
    
//    val sample: Array[(Vector, PartitionIndex)] = elementsWithPartitions.takeSample(false, 2000)
//    val sampleP: RDD[(Vector, PartitionIndex)] = sc.parallelize(sample)

//    val cartesianRDD: RDD[((Vector, PartitionIndex), (Vector, PartitionIndex))] = sampleP.cartesian(sampleP)
//
//    val mapToVectorPair: (((Vector, PartitionIndex), (Vector, PartitionIndex))) => (Vector, Vector) = elem => (elem._1._1, elem._2._1)
//    val mustlinkSource: RDD[(Vector, Vector)] = cartesianRDD.filter(elem => elem._1._2 == elem._2._2).map( mapToVectorPair ).persist(StorageLevel.MEMORY_AND_DISK)
//    logWarning(s"Mustlink source: ${mustlinkSource.count()}");
//    val cannotlinkSource: RDD[(Vector, Vector)] = cartesianRDD.filter(elem => elem._1._2 != elem._2._2).map( mapToVectorPair ).persist(StorageLevel.MEMORY_AND_DISK)
//    logWarning(s"Cannotlink source: ${cannotlinkSource.count()}");

    ( result, fMeasure, purity, time )
  }

  test("Small big test") {
    val elements: Seq[(Vector, Int)] = ClusterGenerator.generate(
      EllipsoidClusterDescriptor(Vectors.dense(0, 0), Vectors.dense(4, 4), 1000),
      EllipsoidClusterDescriptor(Vectors.dense(0, 8), Vectors.dense(1, 1), 100))

    runExperiment(  data = elements,
                    punishmentFactors = Seq(1),
                    mustLinkPercentages = 0 until(100, 10),
                    cannotLinkPercentages = 0 until(100, 10),
                    experimentNumber =  5,
                    "small_big")


  }

  def generateConstraints(source: RDD[(Vector, Vector)], elemNumber: Int, percentage: Double): Set[Constraint] = {

    val constaints: Set[Constraint] = source.takeSample(withReplacement = false, num = (elemNumber * percentage/100).toInt ).map( elem => Constraint((elem._1, elem._2), 1)).toSet
    constaints
  }

  def runExperiment( data: Seq[(Vector, Int)],
                     punishmentFactors: TraversableOnce[Double],
                     mustLinkPercentages: TraversableOnce[Int],
                     cannotLinkPercentages: TraversableOnce[Int],
                     experimentNumber: Int,
                     experimentName: String): Unit = {

    val elementsWithPartitions: RDD[(Vector, Int)] = sc.parallelize(data).persist()
    val k = elementsWithPartitions.map(_._2).distinct().count().toInt
    logWarning(s"ClusterNumber: $k")
    val elementsRDD: RDD[Vector] = elementsWithPartitions.keys.persist()

    val format: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd-hhmm")
    val experimentPath = resultFilePath + experimentName + "_" + format.format(new Date()) + "/"

    new File(experimentPath).mkdir()

    val ( kMeansResult, fMeasure, purity, time ) = calculateKMeansBaseLine(k, elementsWithPartitions, elementsRDD)

    writeClusteringResultToFile(kMeansResult.toLocalIterator, experimentPath + "baseline.csv")

    val base: (Double, Double, Double, Double, Double, Long) = (-1, -1, -1, fMeasure, purity, time)

    val resultFileName = "result.csv"
    val baseLine: String = convertProductToIndexedStringSet(base).mkString(";")
    logInfo(s"Base: $baseLine")
    scala.tools.nsc.io.File(experimentPath + resultFileName).writeAll(baseLine + "\n")

    val elemNumber: Int = elementsRDD.count().toInt

    var cannotLinks: Set[Constraint] = Set.empty
    while(cannotLinks.size < elemNumber) {
      val sample: Array[(Vector, ClusterIndex)] = elementsWithPartitions.takeSample(false, 100)

      val elem = sample(0)
      sample.foreach( e => {
        if(elem._2 != e._2) {
          cannotLinks += Constraint((elem._1, e._1) )
          println(s"Cannotlinks: ${cannotLinks.size}")
        }
      })


    }

    var mustLinks: Set[Constraint] = Set.empty
    while(mustLinks.size < elemNumber) {
      val sample: Array[(Vector, ClusterIndex)] = elementsWithPartitions.takeSample(false, 1000)

      val elem = sample(0)
      sample.foreach( e => {
        if(elem._2 == e._2 && elem._1 != e._1) {
          mustLinks += Constraint((elem._1, e._1) )
          println(s"Mustlinks: ${mustLinks.size}")
        }
      })


    }

    for(  punishmentFactor <- punishmentFactors;
          mustlinkPercentage <- mustLinkPercentages;
          cannotlinkPercentage <- cannotLinkPercentages;
          experimentIndex <- 0 until experimentNumber) {


      logWarning(s"Parameters: punishment = $punishmentFactor, mustlink = $mustlinkPercentage, cannotlink = $cannotlinkPercentage ")


      val stopwatch: Stopwatch = new Stopwatch().start()
      val actualMustLinks: Set[Constraint] = mustLinks.take(elemNumber * mustlinkPercentage / 100)
      println(s"Mustlinks: ${actualMustLinks.size}")
      val actualCannotLinks: Set[Constraint] = cannotLinks.take(elemNumber * cannotlinkPercentage / 100)
      println(s"Cannotlinks: ${actualCannotLinks.size}")
      val pckMeansModel: KMeansModel = PCKMeans.train(elementsRDD, k, 10000, actualMustLinks, actualCannotLinks, punishmentFactor )
      val time: Long = stopwatch.stop.elapsed(TimeUnit.MILLISECONDS)

      val resultRDD: RDD[(Vector, PartitionIndex, ClusterIndex)] = elementsWithPartitions.map(elem => (elem._1, elem._2, pckMeansModel.predict(elem._1))).persist()
      writeClusteringResultToFile(resultRDD.toLocalIterator, s"${experimentPath}${experimentName}_${punishmentFactor}_${mustlinkPercentage}_${cannotlinkPercentage}_$experimentIndex.csv")

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
                    punishmentFactors = Seq(1),
                    mustLinkPercentages = 0 until(100, 10),
                    cannotLinkPercentages = 0 until(100, 10),
                    experimentNumber =  5,
                    "ellipsoid")


  }
  
  def normalize(data: Seq[(Vector, Int)]): Seq[(Vector, Int)] = {
    
    val dimension: ClusterIndex = data.head._1.size
    
    val minCoord: Array[Double] = Array.fill(dimension)(Double.PositiveInfinity)
    val maxCoord: Array[Double] = Array.fill(dimension)(Double.NegativeInfinity)
    
    data.foreach( elem => {
      val coordinates = elem._1.toArray
      val zipped = coordinates.zip(minCoord).zip(maxCoord).zipWithIndex
      
      zipped.foreach( elem => {
        val index = elem._2
        val min = elem._1._1._2
        val max = elem._1._2
        val data = elem._1._1._1
        
        if(min > data) {
          minCoord(index) = data
        }
        if(max < data) {
          maxCoord(index) = data
        }
        
      })
    })
    logWarning("Min: " + minCoord.mkString(";"))
    logWarning("Max: " + maxCoord.mkString(";"))
    val minVector = Vectors.dense(minCoord)
    val diff = maxCoord.zip(minCoord).map( elem => elem._1 - elem._2)
    logWarning("Diff: " + diff.mkString(";"))


    data.map( elem => {
      BLAS.axpy(-1, minVector, elem._1)
      val normalized: Array[Double] = elem._1.toArray.zip(diff).map(elem => elem._1 / elem._2)
      (Vectors.dense(normalized), elem._2)
    })
    
  }

  test("Letters") {

    val lines: Iterator[String] = Source.fromFile("/Users/cskassai/Egyetem/Diplomaterv/letter-recognition.data.txt").getLines()


    val index: Iterator[(String)] = lines.zipWithIndex

      .filter(e => {
      (e._2 % 2) == 0
    }).map(_._1)


    val elements = lines.map( line => {
      val split = line.split(",")
      val coordinates: Array[Double] = split.tail.map(_.toDouble)

      val vector = Vectors.dense(coordinates)
      val cluster = split.head.charAt(0).toInt

      (vector, cluster)
    })
//      .filter(e => {
//      (e._2 % 2) == 0
//    })
      .toSeq

//    val map : MutableMap[Int, (Vector, Int)] = MutableMap.empty
//
//    elements.foreach( elem => {
//      val cluster = elem._2
//      val (sum, count) = map.get(cluster).getOrElse((Vectors.zeros(16), 0))
//
//      val c = count + 1
//
//      BLAS.axpy(1, elem._1, sum)
//
//      map += (cluster -> (sum, c))
//
//    })
//
//    val centers: Map[ClusterIndex, Vector] = map.mapValues(elem => {
//      val sum = elem._1
//      val count = elem._2
//      Vectors.dense(sum.toArray.map(_ / count))
//
//    })
//    centers.toSeq.foreach( elem =>
//      println(s"""${elem._1}: ${elem._2.toArray.mkString(";")}""")
//    )

//    val pcenters: RDD[Vector] = sc.parallelize(centers.values.toSeq)
//    val distances: Array[Double] = pcenters.cartesian(pcenters).map({
//      elem => {
//        val distance = elem._1.toArray.zip(elem._2.toArray).map(e => Math.abs(e._1 - e._2)).sum
//        val dis = Math.sqrt(distance)
//        dis
//      }
//
//    }).distinct().collect()
//    val sorted: Array[Double] = distances.sorted
//    println(sorted.max)
//    println(sorted.min)
//    println(sorted.sum/sorted.length)
//    println(sorted.mkString(";"))
//
//    val model: KMeansModel = new KMeansModel(centers.values.toArray)
//
//    val clustering: (RDD[(Vector, ClusterIndex, ClusterIndex)], Double, Double) = validateClustering(sc.parallelize(elements.map(_._1)), sc.parallelize(elements), model)
//
//    println(clustering._2)
//    println(clustering._3)
    
//    val normalizedElements = normalize(elements)
//
//    val map =sc.parallelize(normalizedElements.map(elem => (elem._2, elem._2 * 2)))
//    val purity: (Double, Seq[(ClusterIndex, Double)]) = ClusteringValidator.calculatePurity(map)
//    logWarning(s"Purity ${purity._1}")
//
//    val f = ClusteringValidator.calculateFMeasure(map)
//    logWarning(s"FMeasure ${f._1}")
//    normalizedElements.foreach(elem => logWarning(elem._1.toArray.mkString(",") + ":" + elem._2))
//
    runExperiment(  data = elements,
      punishmentFactors = Seq(50),
      mustLinkPercentages = 0 until(105, 10),
      cannotLinkPercentages = 0 until(105, 10),
      experimentNumber =  1,
      "letters")


  }


}
