package org.apache.spark.mllib.clustering

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.clustering.Constraint
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Created by cskassai on 04/11/15.
 */
class PCKMeansInitializatorSuite extends SparkFunSuite with MLlibTestSparkContext {

  val dataList = List(  Vectors.dense(3, 1),
                        Vectors.dense(5, 3),
                        Vectors.dense(2, 4),
                        Vectors.dense(9, 4),
                        Vectors.dense(9, 5),
                        Vectors.dense(6, 5),
                        Vectors.dense(6, 6),
                        Vectors.dense(5, 6),
                        Vectors.dense(2, 9),
                        Vectors.dense(2, 10),
                        Vectors.dense(1, 12),
                        Vectors.dense(3, 11),
                        Vectors.dense(6, 11),
                        Vectors.dense(9, 9),
                        Vectors.dense(9, 10)
  )

  val cannotLinkConstraints = Set(  Constraint((Vectors.dense(9, 4), Vectors.dense(9, 5))),
                                    Constraint((Vectors.dense(5, 6), Vectors.dense(2, 9))),
                                    Constraint((Vectors.dense(6, 6), Vectors.dense(6, 11))),
                                    Constraint((Vectors.dense(3, 11), Vectors.dense(6, 11))),
                                    Constraint((Vectors.dense(6, 11), Vectors.dense(9, 10))))

  val mustLinkConstraints = Set(  Constraint((Vectors.dense(6, 5), Vectors.dense(6, 6))),
                                  Constraint((Vectors.dense(5, 6), Vectors.dense(6, 6))),
                                  Constraint((Vectors.dense(2, 10), Vectors.dense(1, 12))),
                                  Constraint((Vectors.dense(2, 10), Vectors.dense(2, 9))),
                                  Constraint((Vectors.dense(3, 11), Vectors.dense(2, 10))),
                                  Constraint((Vectors.dense(9, 9), Vectors.dense(9, 10))))


  test( "collectVectorsFromConstraints" ) {
    val constraintVectors = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize(mustLinkConstraints)
    assert(constraintVectors.size == 9)
  }

  test( "collectVectorsFromConstraints empty input" ) {
    val constraintVectors = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize(Set.empty)
    assert(constraintVectors.size == 0)
  }

  test( "filter constrained elements" ) {
    val mustLinks = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize( mustLinkConstraints )
    val cannotLinks = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize( cannotLinkConstraints )

    val data = sc.parallelize(dataList).map(new VectorWithNorm(_))
    val filtered = PCKMeansInitializator.filterContrainedElementsFromData( data, cannotLinks, mustLinks )
    val filteredData = filtered.collect()

    assert(filteredData.size == 3)
    assert(filteredData.contains(new VectorWithNorm(Vectors.dense(3, 1))))
    assert(filteredData.contains(new VectorWithNorm(Vectors.dense(5, 3))))
    assert(filteredData.contains(new VectorWithNorm(Vectors.dense(2, 4))))
  }

  test( "create components from mustlink" ) {
    val mustLinks = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize( mustLinkConstraints )
    val cannotLinks = PCKMeansInitializator.collectVectorsFromConstraintsAndNormalize( cannotLinkConstraints )

    val (vectorVertexMap, vertexIdToVectorMap) = PCKMeansInitializator.indexConstrainedVectors( mustLinks, cannotLinks )

    assert(vectorVertexMap.size == 12)
    assert(vertexIdToVectorMap.size == 12)


    val components = PCKMeansInitializator.calculateMustLinkComponents( mustLinkConstraints, vectorVertexMap, vertexIdToVectorMap )(sc)

    components.foreach(println)

    assertGrouping( Seq(
                          Seq(VectorWithNorm.fromCoordinates(3, 11), VectorWithNorm.fromCoordinates(1, 12), VectorWithNorm.fromCoordinates(2, 9), VectorWithNorm.fromCoordinates(2, 10)),
                          Seq(VectorWithNorm.fromCoordinates(6, 5), VectorWithNorm.fromCoordinates(5, 6), VectorWithNorm.fromCoordinates(6, 6)),
                          Seq(VectorWithNorm.fromCoordinates(9, 9), VectorWithNorm.fromCoordinates(9, 10)),
                          Seq(VectorWithNorm.fromCoordinates(9, 4)),
                          Seq(VectorWithNorm.fromCoordinates(9, 5)),
                          Seq(VectorWithNorm.fromCoordinates(6, 11))),
                    {components.get(_).get})
  }


  test(" full init ") {
    val (_, _, _, centers) = PCKMeansInitializator.init(5, 1, sc.parallelize(dataList), mustLinkConstraints, cannotLinkConstraints)

    centers.foreach( vectorWithNorm => println(vectorWithNorm.vector))
  }
  
  
  test( " calculating component level cannot link constraints ") {


    val a = 4l
    val d = 5l
    val c = 6l
    val b = 3l
    val e = 2l
    val f = 1l
    val vectorComponentMap = Map(VectorWithNorm.fromCoordinates(9, 4) -> f,
                                VectorWithNorm.fromCoordinates(9, 5) -> e,
                                VectorWithNorm.fromCoordinates(6, 5) -> b,
                                VectorWithNorm.fromCoordinates(6, 6) -> b,
                                VectorWithNorm.fromCoordinates(5, 6) -> b,
                                VectorWithNorm.fromCoordinates(2, 9) -> a,
                                VectorWithNorm.fromCoordinates(2, 10) -> a,
                                VectorWithNorm.fromCoordinates(1, 12) -> a,
                                VectorWithNorm.fromCoordinates(3, 11) -> a,
                                VectorWithNorm.fromCoordinates(6, 11) -> d,
                                VectorWithNorm.fromCoordinates(9, 9) -> c,
                                VectorWithNorm.fromCoordinates(9, 10) -> c)
    
    val componentLevelCannotLinks: Map[Long, Set[Long]] = PCKMeansInitializator.calculateComponentLevelCannotLinks(cannotLinkConstraints, vectorComponentMap)

    componentLevelCannotLinks.foreach( entry => println(s"""${entry._1} => [${entry._2.mkString(", ")}]"""))

    val aCannotLinks = componentLevelCannotLinks.get(a).get
    assert(aCannotLinks.size == 2)
    assert(aCannotLinks.contains(d))
    assert(aCannotLinks.contains(b))

    val bCannotLinks = componentLevelCannotLinks.get(b).get
    assert(bCannotLinks.size == 2)
    assert(bCannotLinks.contains(d))
    assert(bCannotLinks.contains(a))

    val cCannotLinks = componentLevelCannotLinks.get(c).get
    assert(cCannotLinks.size == 1)
    assert(cCannotLinks.contains(d))

    val dCannotLinks = componentLevelCannotLinks.get(d).get
    assert(dCannotLinks.size == 3)
    assert(dCannotLinks.contains(a))
    assert(dCannotLinks.contains(b))
    assert(dCannotLinks.contains(c))

    val eCannotLinks = componentLevelCannotLinks.get(e).get
    assert(eCannotLinks.size == 1)
    assert(eCannotLinks.contains(f))

    val fCannotLinks = componentLevelCannotLinks.get(f).get
    assert(fCannotLinks.size == 1)
    assert(fCannotLinks.contains(e))

  }

  def assertGrouping ( expectedClusters: Seq[Seq[VectorWithNorm]], oracle: VectorWithNorm => Long ): Unit = {
    for( expectedCluster <- expectedClusters ) {
      val predictedClusters: Map[Long, Seq[VectorWithNorm]] = expectedCluster.groupBy(oracle)
      assert(predictedClusters.keySet.size == 1)
    }
  }

}
