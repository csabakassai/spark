package org.apache.spark.mllib.clustering

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.clustering.KMeans._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.slf4j.Logger
import scala.collection.mutable.{Map => MutableMap}

/**
 * Created by cskassai on 19/10/15.
 */
class PCKMeansSuite extends SparkFunSuite with MLlibTestSparkContext {

  val cluster1 = Seq(
    //Cluster 1
    Vectors.dense(0.0, 0.0),
    Vectors.dense(0.0, 2.0),
    Vectors.dense(2.0, 0.0),
    Vectors.dense(2.0, 2.0))

  val cluster2 = Seq(
    //Cluster 1
    Vectors.dense(11.0, 0.0),
    Vectors.dense(13.0, 0.0),
    Vectors.dense(11.0, 2.0),
    Vectors.dense(13.0, 2.0))

  val constrainedPoint = Seq (
    Vectors.dense(4.0, 1.0)
  )

  test("two clusters without constraints") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val rdd = sc.parallelize(points, 3)

    for (initMode <- Seq(RANDOM, K_MEANS_PARALLEL)) {
      // Two iterations are sufficient no matter where the initial centers are.
      val model = KMeans.train(rdd, k = 2, maxIterations = 8, runs = 1, initMode)

      log.error(s"""InitMode: ${initMode}, centers: ${model.clusterCenters.mkString(", ")}""" )
      assertClusters(Seq (cluster1 ++ constrainedPoint, cluster2), model)
    }
  }

  test("two clusters with cannotLink") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val data = sc.parallelize(points, 3)

    val model = PCKMeans.train(data, 2, 4, Set(Constraint((constrainedPoint.apply(0), cluster2.apply(0)), 2.0)), Set.empty)
    log.error(s"""centers: ${model.clusterCenters.mkString(", ")}""" )
    assertClusters(Seq (cluster1 ++ constrainedPoint, cluster2), model)

  }

  test("two clusters with mustLink") {

    val points = cluster1 ++ cluster2 ++ constrainedPoint

    val data = sc.parallelize(points, 3)

    // Two iterations are sufficient no matter where the initial centers are.
    val model = PCKMeans.train(data, 2, 4, Set.empty, Set(Constraint((constrainedPoint.apply(0), cluster1.apply(0)), 2.0)))
    log.error(s"""centers: ${model.clusterCenters.mkString(", ")}""" )
    assertClusters(Seq (cluster1 ++ constrainedPoint, cluster2), model)

  }


  def assertClusters ( expectedClusters: Seq[Seq[Vector]], model: KMeansModel): Unit = {
    for( expectedCluster <- expectedClusters ) {
      val predictedClusters: Map[Int, Seq[Vector]] = expectedCluster.groupBy({model.predict(_)})
      assert(predictedClusters.keySet.size == 1)
    }
  }



}
