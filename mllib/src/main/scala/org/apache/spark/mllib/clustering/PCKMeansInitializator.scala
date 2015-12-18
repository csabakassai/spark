package org.apache.spark.mllib.clustering

import org.apache.spark.graphx._
import org.apache.spark.mllib.clustering.PCKMeans.{ClusterCenter, ClusterIndex}
import org.apache.spark.mllib.linalg.{BLAS, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}
import org.github.jamm.MemoryMeter

/**
*  Created by cskassai on 03/11/15.
*/

object PCKMeansInitializator extends Logging {

  type ComponentId = Long


  def preProcessData (data : RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]): ( RDD[VectorWithNorm], Map[VectorWithNorm, ComponentId], Map[ComponentId, Seq[VectorWithNorm]], Map[ComponentId, Set[ComponentId]])  = {

    logTrace(s"""Mustlink constrains: ${mustLinkConstraints.mkString("\n")}""")
    logTrace(s"""Cannotlink constraints: ${cannotLinkConstraints.mkString("\n")}""")

    val mustLinkConstraintVectorSet: Set[VectorWithNorm] = collectVectorsFromConstraintsAndNormalize( mustLinkConstraints )
    logDebug(s"${mustLinkConstraintVectorSet.size} elem has mustlink constraint")

    val cannotLinkConstraintVectorSet: Set[VectorWithNorm] = collectVectorsFromConstraintsAndNormalize( cannotLinkConstraints )
    logDebug(s"${cannotLinkConstraintVectorSet.size} elem has cannotlink constraint")

    val normalizedData : RDD[VectorWithNorm] = data.map( new VectorWithNorm(_))

    val filteredData: RDD[VectorWithNorm] = filterConstrainedElementsFromData( normalizedData, mustLinkConstraintVectorSet, cannotLinkConstraintVectorSet )

    val (elemToVertexIdMap, vertexIdToElemMap) = indexConstrainedVectors( mustLinkConstraintVectorSet, cannotLinkConstraintVectorSet )

    val (elemToComponentIdMap, elementsByComponents) = calculateMustLinkComponents( mustLinkConstraints, elemToVertexIdMap, vertexIdToElemMap )(data.context)

    val cannotLinkComponentMap = calculateComponentLevelCannotLinks(cannotLinkConstraints, elemToComponentIdMap)


    (filteredData, elemToComponentIdMap, elementsByComponents, cannotLinkComponentMap)
  }


  def collectVectorsFromConstraintsAndNormalize(contraints: Set[Constraint]) : Set[VectorWithNorm] = {
    val constraintVectorSet: Set[Vector] = contraints flatMap { constraint: Constraint => Set( constraint.elements._1, constraint.elements._2)}
    val normalizedContraintVectorSet = constraintVectorSet.map( new VectorWithNorm(_) )
    normalizedContraintVectorSet
  }

  def filterConstrainedElementsFromData (data : RDD[VectorWithNorm], mustLinkConstraintElements: Set[VectorWithNorm], cannotLinkConstraintElements: Set[VectorWithNorm]): RDD[VectorWithNorm] = {

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
                                 (implicit sc: SparkContext): (Map[VectorWithNorm, Long], Map[ComponentId, Seq[VectorWithNorm]]) = {

    val constraintVertices = sc.parallelize(vertexIdToVectorMap.toSeq)

    val edges: Set[Edge[Double]] = mustLinkConstraints.map({ c: Constraint => Edge(vectorToIdMap.get(new VectorWithNorm(c.elements._1)).get, vectorToIdMap.get(new VectorWithNorm(c.elements._2)).get, c.weight) })
    // Create an RDD for edges
    val constraintsRDD: RDD[Edge[Double]] = sc.parallelize(edges.toSeq)

    val graph = Graph(constraintVertices, constraintsRDD)
    val componentsGraph: Graph[VertexId, Double] = graph.connectedComponents()

    val componentVertices: VertexRDD[VertexId] = componentsGraph.vertices

    val vectorComponentList: List[(VectorWithNorm, VertexId)] = componentVertices.map( { rddElem :(VertexId, VertexId) => (vertexIdToVectorMap.get(rddElem._1).get, rddElem._2)}).collect().toList

    val vectorComponentMap = Map(vectorComponentList: _*)
    logWarning(vectorComponentMap.toString())

    ( vectorComponentMap, vectorComponentList.groupBy(_._2).mapValues( _.map(_._1)))
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

  def calculateCenters(k: Int, runs: Int, filteredData : RDD[VectorWithNorm], vectorComponentMap: Map[VectorWithNorm, Long], cannotLinkComponentMap: Map[Long, Set[Long]]) : Map[ClusterIndex, ClusterCenter] = {

    val mustLinkComponents = vectorComponentMap.toSeq.groupBy( _._2 ).toSeq.filter( elem => elem._2.size > 1)

    logWarning(s"${mustLinkComponents.size} mustlinkcomponents exists")

    val sortedComponents = mustLinkComponents.sortBy( elem => elem._2.size )

    val biggestKComponent: Seq[Seq[VectorWithNorm]] = sortedComponents.slice(0, k).map(_._2.map(_._1))


    def calculateVectorCenter(vectorWithNorms  : Seq[VectorWithNorm]): Vector = {
      val vectors: Seq[Vector] = vectorWithNorms.map(_.vector)
      val sum = vectors.reduce(KMeansUtil.addVectors)
      logDebug(s"Sum: $sum")
      BLAS.scal( 1.0 / vectors.size, sum)
      sum

    }
    val biggestKComponentCenter = biggestKComponent.map(calculateVectorCenter).toArray


    val mustLinkComponentCount: PartitionID = biggestKComponentCenter.length
    val randomCenters: Array[Vector] = filteredData.takeSample(withReplacement = false, num = k - mustLinkComponentCount).map(_.vector)

    val centers : Array[VectorWithNorm] = new Array(k)


    for (j <- 0 to k - 1 ) {
      if(j < mustLinkComponentCount) {
        centers(j) = new VectorWithNorm(biggestKComponentCenter(j))
      } else {
        centers(j) = new VectorWithNorm(randomCenters(j - mustLinkComponentCount ))
      }
    }



    centers.zipWithIndex.map( elem => (elem._2, elem._1)).toMap
  }


  def init(k: Int, runs: Int, data : RDD[Vector], mustLinkConstraints: Set[Constraint], cannotLinkConstraints: Set[Constraint]):
            (RDD[VectorWithNorm], Map[VectorWithNorm, VertexId], Map[ComponentId, Seq[VectorWithNorm]], Map[VertexId, Set[VertexId]], Map[ClusterIndex, ClusterCenter]) = {

    val (filteredData, vectorComponentMap, elementsByComponents, cannotLinkComponentMap) = preProcessData(data, mustLinkConstraints, cannotLinkConstraints)

    val centers: Map[ClusterIndex, ClusterCenter] = calculateCenters(k, runs, filteredData, vectorComponentMap, cannotLinkComponentMap)


    (filteredData, vectorComponentMap, elementsByComponents, cannotLinkComponentMap, centers)
  }


}
