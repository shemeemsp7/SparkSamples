package com.sps.sample

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

object CollaborativeFilter {
  
  def main(args: Array[String]): Unit = {
    
   val conf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
   val sc = new SparkContext(conf) 
   
    val data = sc.textFile("src/main/resources/test.data")
    val ratings = data.map(_.split(',') match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })

  // Build the recommendation model using ALS
  val rank = 10
  val numIterations = 10
  val model = ALS.train(ratings, rank, numIterations, 0.01)

  // Evaluate the model on rating data
  val usersProducts = ratings.map { case Rating(user, product, rate) =>
    (user, product)
  }
  val predictions =
    model.predict(usersProducts).map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }
  val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }.join(predictions)
  
  ratesAndPreds.foreach(data => {
    println(data);
  })
  val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
    val err = (r1 - r2)
    err * err
  }.mean()
  println("Mean Squared Error = " + MSE)
  
  // Save and load model
  model.save(sc, "target/tmp/myCollaborativeFilter")
  val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
  
  }
}