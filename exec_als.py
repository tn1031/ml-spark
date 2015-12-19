#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join
import numpy as np

from pyspark import SparkConf, SparkContext


def parseRating(line):
  fields = line.strip().split("::")
  return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
  fields = line.strip().split("::")
  return int(fields[0]), fields[1]

if __name__ == '__main__':
  conf = SparkConf() \
    .setAppName("MovieLensALS") \
    .set("spark.executor.memory", "2g")
  sc = SparkContext(conf=conf)

  movieLensHomeDir = "ml-1m"
  ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)
  movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())
  
  
  n_ratings = ratings.count()
  n_user = ratings.values().map(lambda r: r[0]).distinct().count()
  n_item = ratings.values().map(lambda r: r[1]).distinct().count()
  
  print "Got %d ratings from %d users on %d movies." % (n_ratings, n_user, n_item)
  
  numPartitions = 4
  training = ratings.filter(lambda x: x[0] < 6) \
    .values() \
    .repartition(numPartitions) \
    .cache()

  validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
    .values() \
    .repartition(numPartitions) \
    .cache()

  test = ratings.filter(lambda x: x[0] >= 8).values().cache()

  n_training = training.count()
  n_validation = validation.count()
  n_test = test.count()

  print "Training: %d, validation: %d, test: %d" % (n_training, n_validation, n_test)
  
  """
  Training
  """
  users = training.map(lambda r: r[0]).distinct()
  items = training.map(lambda r: r[1]).distinct()

  rank = 5
  lmd = 0.1
  epochs = 3

  X = users.map(lambda u: (u, np.random.uniform(size=rank).reshape(rank, 1)))
  Y = items.map(lambda i: (i, np.random.uniform(size=rank).reshape(rank, 1)))

  for epoch in xrange(epochs):
    # update user matrix X
    joined = training.map(lambda x: (x[1], (x[0], x[2]))).join(Y)
    term1 = joined.map(lambda x: (x[1][0][0], np.dot(x[1][1], np.transpose(x[1][1])))) \
      .reduceByKey(lambda x, y: x+y) \
      .mapValues(lambda x: x + np.identity(rank))
    term2 = joined.map(lambda x: (x[1][0][0], x[1][0][1]  * x[1][1])) \
      .reduceByKey(lambda x, y: x+y)
    X = term1.join(term2) \
      .mapValues(lambda x: np.linalg.solve(x[0], x[1]))
  
    # update item matrix Y
    joined = training.map(lambda x: (x[0], (x[1], x[2]))).join(X)
    term1 = joined.map(lambda x: (x[1][0][0], np.dot(x[1][1], np.transpose(x[1][1])))) \
      .reduceByKey(lambda x, y: x+y) \
      .mapValues(lambda x: x + np.identity(rank))
    term2 = joined.map(lambda x: (x[1][0][0], x[1][0][1]  * x[1][1])) \
      .reduceByKey(lambda x, y: x+y)
    Y = term1.join(term2) \
      .mapValues(lambda x: np.linalg.solve(x[0], x[1]))

    # calc RMSE
    rmse = np.sqrt(
      training.map(lambda x: (x[0], (x[1], x[2]))) \
        .join(X) \
        .map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1]))) \
        .join(Y) \
        .map(lambda x: ((x[0], x[1][0][0]), (x[1][0][1], x[1][0][2], x[1][1]))) \
        .mapValues(lambda x: (x[0] - np.dot(np.transpose(x[1]), x[2]))**2) \
        .values() \
        .reduce(lambda x, y: x + y) \
      / n_training)

    print "epoch: %d, rmse: %f" % (epoch, rmse)

  # clean up
  sc.stop()
