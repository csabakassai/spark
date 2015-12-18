package org.apache.spark.mllib.clustering

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics
import org.apache.spark.rdd.RDD

import scala.Predef
import scala.collection.Map

/**
*  Created by cskassai on 27/11/15.
*/
object ClusteringValidator {

  type PartitionIndex = Int

  type ClusterIndex = Int


  def calculateFMeasure( data: RDD[(PartitionIndex, ClusterIndex)]): (Double, Seq[(ClusterIndex, Double, Double, Double)] ) = {

    val contingencyTable: Map[(PartitionIndex, ClusterIndex), Long] = data.countByValue()

    val contingencyTableByCluster: Map[ClusterIndex, Seq[(PartitionIndex, Long)]] = contingencyTable.toSeq.groupBy(_._1._2).mapValues( _.map( elem => (elem._1._1, elem._2)))

    val elemNumberByCluster: Map[ClusterIndex, Long] = contingencyTableByCluster.mapValues(_.map(_._2).sum)

    val contingencyTableByPartition: Map[PartitionIndex, Seq[(ClusterIndex, Long)]] = contingencyTable.toSeq.groupBy(_._1._1).mapValues( _.map( elem => (elem._1._2, elem._2)))
    val elemNumberByPartition: Map[PartitionIndex, Long] = contingencyTableByPartition.mapValues(_.map(_._2).sum)

    val maxPartitions: Seq[(ClusterIndex, PartitionIndex)] = contingencyTableByCluster.mapValues( _.maxBy(_._2)).toSeq.map( elem => (elem._1, elem._2._1))

    def getRecall(cluster: ClusterIndex, partition: PartitionIndex): Double = {
      val n = contingencyTable.get((partition, cluster)).get
      val partitionElemNumber = elemNumberByPartition.get(partition).get

      n/partitionElemNumber.toDouble
    }

    def getPrecision(cluster: ClusterIndex, partition: PartitionIndex): Double = {
      val n = contingencyTable.get((partition, cluster)).get
      val clusterElemNumber = elemNumberByCluster.get(cluster).get
      n/clusterElemNumber.toDouble
    }

    val precisionRecallAndFMeasureByCluster: Seq[(ClusterIndex, Double, Double, Double)] = maxPartitions.map( e => (e._1, getPrecision(e._1, e._2), getRecall(e._1, e._2))).map(e => (e._1, e._2, e._3, harmonicMean(e._2, e._3)))

    val fMeasure: Double = precisionRecallAndFMeasureByCluster.map(_._4).sum / precisionRecallAndFMeasureByCluster.size

    (fMeasure, precisionRecallAndFMeasureByCluster)
  }

  def harmonicMean( data: Double*): Double = {
    if (data.contains(0.0))
      return Double.NaN
    data.length / data.foldLeft(0.0)((s, t) => s + 1.0 / t)
  }

  def calculatePurity( data: RDD[(PartitionIndex, ClusterIndex)]): (Double, Seq[(Int, Double)] ) = {

    val elemNumber: Long = data.count()

    val contingencyTable: Map[(PartitionIndex, ClusterIndex), Long] = data.countByValue()
    
    val contingencyTableByCluster: Map[ClusterIndex, Seq[Long]] = contingencyTable.toSeq.groupBy(_._1._2).mapValues( _.map( _._2))
    
    val purityAndElemNumberByCluster: Seq[(ClusterIndex, (Double, Long))] = contingencyTableByCluster.mapValues( elem => {
      val max = elem.max
      val sum = elem.sum
      val purity = max/(sum.toDouble)
      (purity, sum)
    }).toSeq

    val purity: Double = purityAndElemNumberByCluster.map( elem => elem._2._1 * elem._2._2 / elemNumber.toDouble).sum

    val purityByCluster: Seq[(ClusterIndex, Double)] = purityAndElemNumberByCluster.map(elem => (elem._1, elem._2._1))

    (purity, purityByCluster)
  }

}
