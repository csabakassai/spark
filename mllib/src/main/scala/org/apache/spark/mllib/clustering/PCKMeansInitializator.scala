package org.apache.spark.mllib.clustering

import org.apache.spark.SparkContext
import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.KMeansUtil
import org.apache.spark.mllib.clustering.VectorWithNorm
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors, BLAS}

/**
 * Created by cskassai on 03/11/15.
 */
object PCKMeansInitializator {

  type ComponentId = Long


  def preProcessData (data : RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): ( RDD[VectorWithNorm], Map[VectorWithNorm, ComponentId], Map[ComponentId, Set[ComponentId]], Map[Long, VectorWithNorm])  = {

    val mustLinkConstraintVectorSet: Set[VectorWithNorm] = collectVectorsFromConstraintsAndNormalize( mustLinkConstraints )

    val cannotLinkConstraintVectorSet: Set[VectorWithNorm] = collectVectorsFromConstraintsAndNormalize( cannotLinkConstraints )

    val normalizedData : RDD[VectorWithNorm] = data.map( new VectorWithNorm(_))

    val filteredData: RDD[VectorWithNorm] = filterContrainedElementsFromData( normalizedData, mustLinkConstraintVectorSet, cannotLinkConstraintVectorSet )

    val (vectorVertexMap, vertexIdToVectorMap) = indexConstrainedVectors( mustLinkConstraintVectorSet, cannotLinkConstraintVectorSet )

    val vectorComponentMap = calculateMustLinkComponents( mustLinkConstraints, vectorVertexMap, vertexIdToVectorMap )(data.context)

    val cannotLinkComponentMap = calculateComponentLevelCannotLinks(cannotLinkConstraints, vectorComponentMap)


    (filteredData, vectorComponentMap, cannotLinkComponentMap, vertexIdToVectorMap)
  }


  def collectVectorsFromConstraintsAndNormalize(contraints: Set[Constraint]) : Set[VectorWithNorm] = {
    val constraintVectorSet: Set[Vector] = contraints flatMap { constraint: Constraint => Set( constraint.elements._1, constraint.elements._2)}
    val normalizedContraintVectorSet = constraintVectorSet.map( new VectorWithNorm(_) )
    normalizedContraintVectorSet
  }

  def filterContrainedElementsFromData (data : RDD[VectorWithNorm], mustLinkConstraintElements: Set[VectorWithNorm], cannotLinkConstraintElements: Set[VectorWithNorm]): RDD[VectorWithNorm] = {

    val filteredData: RDD[VectorWithNorm] = data.filter( vector => ! (cannotLinkConstraintElements.contains(vector) || mustLinkConstraintElements.contains(vector)))
    filteredData
  }

  def indexConstrainedVectors ( mustLinkConstraintElements: Set[VectorWithNorm], cannotLinkConstraintElements: Set[VectorWithNorm] ) : ( Map[VectorWithNorm, Long], Map[Long, VectorWithNorm]) = {

    val contrainedElements = mustLinkConstraintElements ++ cannotLinkConstraintElements

    val indexedConstaintVectorSet = contrainedElements.zip(Stream.from(1).map({ index: Int => index.toLong})).toList

    val inversedIndexedConstaintVectorSet = indexedConstaintVectorSet map { elem => (elem._2, elem._1)}

    val vectorToIdMap = Map(indexedConstaintVectorSet:_*)

    val vertexIdToVectorMap = Map(inversedIndexedConstaintVectorSet:_*)

    (vectorToIdMap, vertexIdToVectorMap)
  }


  def calculateMustLinkComponents(  mustLinkConstraints: Set[Constraint],
                                    vectorToIdMap: Map[VectorWithNorm, Long],
                                    vertexIdToVectorMap: Map[Long, VectorWithNorm])
                                 (implicit sc: SparkContext): Map[VectorWithNorm, Long] = {

    val constraintVertices = sc.parallelize(vertexIdToVectorMap.toSeq)

    val edges: Set[Edge[Double]] = mustLinkConstraints.map({ c: Constraint => Edge(vectorToIdMap.get(new VectorWithNorm(c.elements._1)).get, vectorToIdMap.get(new VectorWithNorm(c.elements._2)).get, c.weight) })
    // Create an RDD for edges
    val constraintsRDD: RDD[Edge[Double]] = sc.parallelize(edges.toSeq)

    val graph = Graph(constraintVertices, constraintsRDD)
    val componentsGraph: Graph[VertexId, Double] = graph.connectedComponents()

    val componentVertices: VertexRDD[VertexId] = componentsGraph.vertices

    val vectorComponentList: List[(VectorWithNorm, VertexId)] = componentVertices.map( { rddElem :(VertexId, VertexId) => (vertexIdToVectorMap.get(rddElem._1).get, rddElem._2)}).collect().toList

    val vectorComponentMap = Map(vectorComponentList: _*)

    vectorComponentMap
  }
  
  def calculateComponentLevelCannotLinks( cannotLinkConstraints: Set[Constraint], vectorComponentMap : Map[VectorWithNorm, Long] ) : Map[Long, Set[Long]] = {

    val cannotLinkComponentSet  = for {
                                      cannotLinkConstraint <- cannotLinkConstraints
                                      elements: (Vector, Vector) = cannotLinkConstraint.elements
                                      firstVector: Vector = elements._1
                                      secondVector: Vector = elements._2
                                      firstComponentId = vectorComponentMap.get(new VectorWithNorm(firstVector)).get
                                      secondComponentId = vectorComponentMap.get(new VectorWithNorm(secondVector)).get
                                    } yield (firstComponentId, secondComponentId)

    val inversCannotLinkComponentSet: Set[(Long, Long)] = cannotLinkComponentSet map { elem => (elem._2, elem._1)}

    val sumCannotLinkComponetSet: Set[(Long, Long)] = cannotLinkComponentSet ++ inversCannotLinkComponentSet

    val cannotLinkComponentMap = sumCannotLinkComponetSet groupBy (_._1) mapValues { componentTupleSeq => componentTupleSeq map { elem => elem._2}}
    
    cannotLinkComponentMap
  }

  def calculateCenters(k: Int, runs: Int, filteredData : RDD[VectorWithNorm], vectorComponentMap: Map[VectorWithNorm, Long], cannotLinkComponentMap: Map[Long, Set[Long]], vertexIdToVectorMap: Map[Long, VectorWithNorm]) : Array[VectorWithNorm] = {

    val mustLinkComponents = vectorComponentMap.toSeq.groupBy( _._2 ).toSeq.filter( elem => elem._2.size > 1)

    val sortedComponents = mustLinkComponents.sortBy( elem => elem._2.size )

    val biggestKComponent: Seq[Seq[VectorWithNorm]] = sortedComponents.slice(0, k-1).map(_._2.map(_._1))

    val dims = filteredData.take(1)(0).vector.size


    def calculateVectorCenter(vectorWithNorms  : Seq[VectorWithNorm]): Vector = {
      val vectors: Seq[Vector] = vectorWithNorms.map(_.vector)
      val sum: Vector = vectors.reduce(KMeansUtil.addVectors)
      println(s"Sum: ${sum}")
      BLAS.scal( 1.0 / vectors.size, sum)
      sum

    }
    val biggestKComponentCenter = biggestKComponent.map(calculateVectorCenter).toArray

    println("Centers")
    biggestKComponentCenter.foreach(println)
    println("=====")

    val mustLinkComponentCount: PartitionID = biggestKComponentCenter.size
    val randomCenters: Array[Vector] = filteredData.takeSample(false, k - mustLinkComponentCount).map(_.vector)

    val centers : Array[VectorWithNorm] = new Array(k)


    for (j <- 0 to k - 1 ) {
      if(j < mustLinkComponentCount) {
        centers(j) = new VectorWithNorm(biggestKComponentCenter(j))
      } else {
        centers(j) = new VectorWithNorm(randomCenters(j - mustLinkComponentCount ))
      }
    }



    centers
  }


  def init(k: Int, runs: Int, data : RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): (RDD[VectorWithNorm], Map[VectorWithNorm, VertexId], Map[VertexId, Set[VertexId]], Array[VectorWithNorm]) = {

    val (filteredData, vectorComponentMap, cannotLinkComponentMap, vertexIdToVectorMap) = preProcessData(data, mustLinkConstraints, cannotLinkConstraints)

    val centers: Array[VectorWithNorm] = calculateCenters(k, runs, filteredData, vectorComponentMap, cannotLinkComponentMap, vertexIdToVectorMap)

    (filteredData, vectorComponentMap, cannotLinkComponentMap, centers)
  }


}
