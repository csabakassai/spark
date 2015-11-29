package org.apache.spark.mllib.clustering

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.Matchers._

/**
*  Created by cskassai on 27/11/15.
*/
class ClusteringValidatorSuite extends SparkFunSuite with MLlibTestSparkContext {
  
  
  test("Purity: Perfect clustering with same index") {
    
    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 1), (2, 2), (1, 1), (1, 1), (3, 3), (2, 2)))
    
    val (purity, purityByCluster) = ClusteringValidator.calculatePurity(data)

    purity shouldBe 1.0 +- 0.0000001

    purityByCluster.foreach( _._2 shouldBe 1.0 +- 0.0000001)
    
  }

  test("Purity: Perfect clustering with different index") {

    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 2), (2, 3), (1, 2), (1, 2), (3, 1), (2, 3)))

    val (purity, purityByCluster) = ClusteringValidator.calculatePurity(data)

    purity shouldBe 1.0 +- 0.0000001

    purityByCluster.foreach( _._2 shouldBe 1.0 +- 0.0000001)

  }

  test("Purity: Not perfect clustering with different index") {

    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (2, 1)))

    val (purity, purityByCluster) = ClusteringValidator.calculatePurity(data)

    purity shouldBe 0.8333333333333333 +- 0.0000001

  }

  test("F-Measure: Perfect clustering with same index") {

    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 1), (2, 2), (1, 1), (1, 1), (3, 3), (2, 2)))

    val (fMeasure, fMeasureByCluster) = ClusteringValidator.calculateFMeasure(data)

    fMeasure shouldBe 1.0 +- 0.0000001

    fMeasureByCluster.foreach( _._2 shouldBe 1.0 +- 0.0000001)

  }

  test("F-Measure : Perfect clustering with different index") {

    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 2), (2, 3), (1, 2), (1, 2), (3, 1), (2, 3)))

    val (fMeasure, fMeasureByCluster) = ClusteringValidator.calculateFMeasure(data)

    logError(fMeasureByCluster.mkString("; "))

    fMeasure shouldBe 1.0 +- 0.0000001


    fMeasureByCluster.foreach( _._2 shouldBe 1.0 +- 0.0000001)

  }

  test("F-Measure: Not perfect clustering with different index") {

    val data: RDD[(Int, Int)] = sc.parallelize(Seq((1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (2, 1)))

    val (fMeasure, fMeasureByCluster) = ClusteringValidator.calculateFMeasure(data)

    fMeasure shouldBe 0.828571428571428 +- 0.0000001

  }

}
