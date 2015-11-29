package org.apache.spark.mllib.clustering

import java.io.File

import breeze.io.CSVWriter
import com.google.protobuf.Descriptors
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import scala.collection.immutable
import scala.math.{Pi, cos, sin, sqrt}

import scala.util.Random

/**
*  Created by cskassai on 22/11/15.
*/

object ClusterGenerator {

  def generate(descriptors: EllipsoidClusterDescriptor*) : Seq[(Vector, Int)] = {


    val elements = for ( descriptor <- descriptors.zipWithIndex;
                         cluster: Seq[Vector] = generateCluster(descriptor._1)
                        ) yield cluster.map( (_, descriptor._2))

    elements.flatten
  }

  def generateCluster(descriptor: EllipsoidClusterDescriptor): Seq[Vector] = {
    Stream.continually(generateOneElem(descriptor)).take(descriptor.elemNumber.toInt)
  }

  def generateOneElem(descriptor: EllipsoidClusterDescriptor): Vector = {
    val rho: Double = Random.nextDouble()
    val phi: Double = Random.nextDouble() * 2 * Pi
    val x = sqrt(rho) * cos(phi)
    val y = sqrt(rho) * sin(phi)

    val ground: Vector = Vectors.dense(x,y)

    val transformed: Vector = Vectors.dense(ground.toArray.zip(descriptor.extent.toArray).map( elem => elem._1 * elem._2 ))

    BLAS.axpy(1, descriptor.center, transformed)

    transformed
  }

  def main (args: Array[String]){


    val elements: Seq[(Vector, Int)] = generate(
      EllipsoidClusterDescriptor(Vectors.dense(0, 0), Vectors.dense(4, 4), 10000),
      EllipsoidClusterDescriptor(Vectors.dense(0, 6), Vectors.dense(1, 1), 10000))
    val toIndexedSeq: immutable.IndexedSeq[immutable.IndexedSeq[String]] = elements.map( elem => elem._1.toArray.map( _.toString ).toIndexedSeq ).toIndexedSeq

    CSVWriter.writeFile(new File("/Users/cskassai/Egyetem/Diplomaterv/clusters_2.csv"), toIndexedSeq)


  }
}


case class EllipsoidClusterDescriptor( center: Vector, extent: Vector, elemNumber: Long)
