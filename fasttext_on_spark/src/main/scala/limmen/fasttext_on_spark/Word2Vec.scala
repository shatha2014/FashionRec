package limmen.fasttext_on_spark

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import java.io.{ BufferedWriter, FileWriter }
import org.apache.log4j.{ Level, LogManager }
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import scala.collection.mutable.HashMap
import scala.util.Random

/**
 * Entry in vocabulary
 */
private case class W2VVocabWord(
  var word: String, //The word itself
  var cn: Int, //Word frequency
  var point: Array[Int], //Path from Root to the word in the Huffman Tree, store every index of non-leaf node
  var code: Array[Int], //Huffman code
  var codeLen: Int //Length of the code
)

/**
 * Word2VecOnSpark, based on implementation from:
 * https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/feature/Word2Vec.scala
 *
 * Original C implementation: https://github.com/tmikolov/word2vec
 *
 * Original paper:
 *
 * @article{DBLP:journals/corr/MikolovSCCD13,
 * author    = {Tomas Mikolov and
 *              Ilya Sutskever and
 *               Kai Chen and
 *                Greg Corrado and
 *                Jeffrey Dean},
 *   title     = {Distributed Representations of Words and Phrases and their Compositionality},
 *   journal   = {CoRR},
 *   volume    = {abs/1310.4546},
 *   year      = {2013}
 * }
 *
 * @author Kim Hammar <kimham@kth.se> <limmen@github>
 */
class Word2Vec extends Serializable {

  //Used to generate random seeds
  private val rand = Random
  //Size of word embedding
  private var vectorSize = 100
  //Learningrate for optimization
  private var learningRate = 0.025
  //Partitions to distribute the training on
  private var numPartitions = 1
  //Number of iterations for optimization
  private var numIterations = 1
  //seed used to for PRNGs for random initialization.
  private var seed = rand.nextLong()
  //Minimum word frequency, used for filtering out un-frequent words
  private var minCount = 5
  //Max length of a given sentence (used to compute contexts of words)
  private var maxSentenceLength = 1000

  /**
   * EXP_TABLE is used as an optimization to make it faster to compute Sigmoids over a large
   * vocabulary (at the expense of more memory requirement).
   * The table holds the values of sigmoid in a given range (-6 -- 6 default)
   * which allows to lookup values of Sigmoid rather than re-computing.
   * This type of table is especially suited for Sigmoids since Sigmoids saturate near 1 and -1
   * and it is enough to store a single y-value for a range of x-values.
   * I.e, the Sigmoid is divided into pieces of values that are stored in the table, to lookup
   * the value of a input x-value, we just need to know which piece (range) it belongs to.
   */
  //Exponential table size, provides 1000 temporary results
  private val EXP_TABLE_SIZE = 1000
  // From -6 (exp^-6 / (exp^-6 + 1)) to 6 (exp^6 / (exp^6 + 1))
  private val MAX_EXP = 6
  //Maximum HUffman Code Length in the tree
  private val MAX_CODE_LENGTH = 40

  //context words from [-window, window]
  //Indicate the Maximum scope of '' in Cbow;
  //'max space between words（w1,w2,p(w1 | w2)）' in Skip-gram
  private var window = 5
  //Training Words Number(Accumulation of Word Frequency)
  private var trainWordsCount = 0L
  // Current Vocabulary Size
  private var vocabSize = 0
  //Transient to denote that these values should not be serialized
  //Word Vocabulary
  @transient private var vocab: Array[W2VVocabWord] = null
  //Vocabulary Hash; Index: Hash Value; Content: Position of Word in Vocab;
  //a[word_hash] = word index in vocab
  @transient private var vocabHash = HashMap.empty[String, Int]
  //Boolean if output should be in raw format or in L2-Normalized format
  private var norm = false
  //Boolean if output from partitions should be averaged (otherwise summed)
  private var average = false
  //Boolean if logging should be verbose
  private var verbose = false

  /**
   * Setters
   */

  /**
   * Sets the maximum length (in words) of each sentence in the input data.
   * Any sentence longer than this threshold will be divided into chunks of
   * up to `maxSentenceLength` size (default: 1000)
   *
   * @param maxSentencelength the sentence lenght to set
   * @return reference to the Word2Vec instance
   */
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(
      maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got ${maxSentenceLength}")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
   * Sets vector size (default: 100).
   *
   * @param vectorSize the vectorSize to set
   * @return reference to the Word2Vec instance
   */
  def setVectorSize(vectorSize: Int): this.type = {
    require(
      vectorSize > 0,
      s"vector size must be positive but got ${vectorSize}")
    this.vectorSize = vectorSize
    this
  }

  /**
   * Sets initial learning rate (default: 0.025).
   *
   * @param learningRate the learningRate to set
   * @return reference to the Word2Vec instance
   */
  def setLearningRate(learningRate: Double): this.type = {
    require(
      learningRate > 0,
      s"Initial learning rate must be positive but got ${learningRate}")
    this.learningRate = learningRate
    this
  }

  /**
   * Sets number of partitions (default: 1). Use a small number for accuracy.
   *
   * @param numPartitions the number of partitons to set
   * @return reference to the Word2Vec instance
   */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(
      numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   *
   * @param numIterations the number of iterations to set
   * @return reference to the Word2Vec instance
   */
  def setNumIterations(numIterations: Int): this.type = {
    require(
      numIterations >= 0,
      s"Number of iterations must be nonnegative but got ${numIterations}")
    this.numIterations = numIterations
    this
  }

  /**
   * Sets random seed (default: a random long integer).
   *
   * @param seed, the seed to set
   * @return reference to the Word2Vec instance
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Sets the window of words (default: 5)
   *
   * @param window the size of the context window to set
   * @return reference to the Word2Vec instance
   */
  def setWindowSize(window: Int): this.type = {
    require(
      window > 0,
      s"Window of words must be positive but got ${window}")
    this.window = window
    this
  }

  /**
   * Sets minCount, the minimum number of times a token must appear
   * to be included in the word2vec model's vocabulary (default: 5).
   *
   * @param minCount the mincount to set
   * @return reference to the Word2Vec instance
   */
  def setMinCount(minCount: Int): this.type = {
    require(
      minCount >= 0,
      s"Minimum number of times must be nonnegative but got ${minCount}")
    this.minCount = minCount
    this
  }

  /**
   * Sets norm, boolean indicating if output should be in L2-normalized form. Default: false
   *
   * @param norm boolean to set
   * @return reference to the Word2Vec instance
   */
  def setNorm(norm: Boolean): this.type = {
    this.norm = norm
    this
  }

  /**
   * Sets average, boolean indicating if output by partitions should be averaged.
   * Otherwise the partitions are summed. Default: false
   *
   * @param average boolean to set
   * @return reference to the Word2Vec instance
   */
  def setAverage(average: Boolean): this.type = {
    this.average = average
    this
  }

  /**
   * Sets verbose, boolean indicating if logging should be verbose.
   * Default: false
   *
   * @param average boolean to set
   * @return reference to the Word2Vec instance
   */
  def setVerbose(verbose: Boolean): this.type = {
    this.verbose = verbose
    this
  }

  /**
   * Learns vocabulary table from input RDD of sentences of strings.
   * The vocabulary produced after this method contains all the words and their counts
   * but lacks the huffman codes. This method also populates the vocabHash.
   *
   * @param dataset RDD input corpus
   */
  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    //Flatten list of sentences into list of words
    val words = dataset.flatMap(x => x)

    //Compute vocabulary by counting occurences of words and filtering based on minCount
    //Furthermore, each word is transformed into its VocabWord format with dummy entries for
    //the huffman codes since the tree is not computed yet. Sort by count.
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => W2VVocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    //Populate vocabulary hash with (Word -> IndexInVocab)
    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    LogHolderW2V.log.info(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  /**
   * Creates the EXP_TABLE of pre-computed exp(x) values.
   *
   * @returns array representing the partitioned EXP table
   */
  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP) // Precompute the exp() table
      expTable(i) = (tmp / (tmp + 1.0)).toFloat // Precompute f(x) = x / (x + 1)
      i += 1
    }
    expTable
  }

  /**
   * Creates Binary Huffman Tree using the Word Counts.
   * Frequent Words will have Short Uniqe Binary Codes.
   * The tree is constructed based on the word-frequency table.
   */
  private def createBinaryTree(): Unit = {
    //holds counts of nodes in the tree (*2 since the tree will be larger than the vocabulary (vocabulary is all leaf nodes)
    val count = new Array[Long](vocabSize * 2 + 1)
    //holds binary representation of words
    val binary = new Array[Int](vocabSize * 2 + 1)
    //holds parentnodes of all nodes, this is essentially the Huffman tree representation
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    //holds a binary huffman code
    val code = new Array[Int](MAX_CODE_LENGTH)
    //path from root to a given leaf-word in the Huffman tree, stores the index of every itermediate node on the path
    val point = new Array[Int](MAX_CODE_LENGTH)

    var a = 0
    //Fill up the initial leaf node counts for all individual words
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    //Fill upp the counts for the parent nodes to a MIN-VALUE (non-vocab entries)
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    //Start with the lowest-frequency words (sorted by count)
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0
    a = 0

    // Following Algorithm Constructs The Huffman tree By Adding One Node At A Time
    while (a < vocabSize - 1) {
      //Find the smallest of the two nodes, min1i, min2i
      //These two nodes will be "merged" into a parent node in the tree
      //Update pos1 and pos2 to prepare for next iteration
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      //Set the count for the merged parent node
      count(vocabSize + a) = count(min1i) + count(min2i)
      //Set the index of the parent node for the two min-nodes that were merged
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      //Set binary representation of left node (right node is already correct, initialized to 0)
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word by traversing the tree
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b) // codes, assign the binary value for each step in the tree
        point(i) = b // Assign the index of each node in the tree path
        i += 1
        b = parentNode(b)
      }
      // Notice that Point Is One Layer Deeper Than Code At The Same Posiiton
      vocab(a).codeLen = i // Not Count for Root
      vocab(a).point(0) = vocabSize - 2 // Reverse Order，Assign First One as Root (2*vocab_size - 2 - vocab_size)
      b = 0
      //Assign the Huffman codes and paths from root to the word in the tree to the previously learned vocabulary
      while (b < i) { // Reverse Order
        vocab(a).code(i - b - 1) = code(b) //No Root, Left Child = 0, Right Child = 1
        vocab(a).point(i - b) = point(b) - vocabSize // Notice Last Element In point Array Is Negative, Is Meaningless
        b += 1
      }
      a += 1
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   *
   * @param dataset an RDD of sentences,
   * each sentence is expressed as an iterable collection of words
   * @return a Word2VecModel
   */
  def train[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {

    //Learn the Vocabulary Frequencies and create placeholder for huffman codes
    learnVocab(dataset)

    //Learn the huffman codes
    createBinaryTree()

    //Obtain reference to the cluster
    val sc = dataset.context

    /**
     * Broadcast read-only variables to all machines in the cluster
     * This procedure is optimized by Spark for read-only access. It is faster than shipping
     * the varaibles with each task in the job.
     */

    //The precomputed EXP table is read-only and all machines need access
    val expTable = sc.broadcast(createExpTable())
    //The learned vocabulary is read-only and all machines need access
    val bcVocab = sc.broadcast(vocab)
    //The vocabularyHash for eficient lookup in vocabulary is read-only and all machines need access
    val bcVocabHash = sc.broadcast(vocabHash)

    try {
      //Call method containing the training logic
      doTrain(dataset, sc, expTable, bcVocab, bcVocabHash)
    } finally {
      //Cleanup after training
      //Destroy broadcast variables, blocks until destroying is complete
      expTable.destroy()
      bcVocab.destroy()
      bcVocabHash.destroy()
    }
  }

  /**
   * Containis the logic for training the word embeddings using skip-gram and
   * Hierarchical softmax.
   * Takes dataset of sentences and broadcast variables as input
   *
   * @param dataset the RDD representing the input corpus
   * @param expTable broadcast variable with precomputed exponentials
   * @param bcVocab broadcast variable with the vocabulary containing words, huffman codes, etc
   * @param bcVocabHash broadcast variable with hashes/indexes of words in vocab
   */
  private def doTrain[S <: Iterable[String]](
    dataset: RDD[S], sc: SparkContext,
    expTable: Broadcast[Array[Float]],
    bcVocab: Broadcast[Array[W2VVocabWord]],
    bcVocabHash: Broadcast[HashMap[String, Int]]) = {
    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int] corresponding to vocabulary indexes
      sentenceIter.flatMap { sentence =>
        // Sentence of words, some of which map to a word index
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }
    //Make sure senteces are partitioned over all machines and cached in memory
    val newSentences = sentences.repartition(numPartitions).cache()
    //PRNG for random initialization of vectors
    val initRandom = Random

    //Make sure indices fit withing Int.MaxValue
    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }
    //syn0 represents concatenated word vectors.
    //Initialize word vectors randomly.
    //Notice that the vectors are represented in a single long flattened vector rather than a matrix
    //To access individual vectors in the long vector, it needs to be sliced or use indices.
    val syn0Global =
      Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    //syn1 represents weights from hidden layer to non-leaf node in huffman tree in Hierarhical softmax
    //No initialization of syn1 here
    //Similar to Syn0Global, the weights are represented as a single long vector which is sliced to access individual weights
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    //Keep track of number of words processed in the skipgram training.
    val totalWordsCounts = numIterations * trainWordsCount + 1
    var alpha = learningRate
    //Initialize global loss
    var globalLoss = 0.0f

    //The actual training iterations
    for (k <- 1 to numIterations) {
      //Create broadcast variables of word vectors and hidden weights in this iteration
      //These will be distributed to each worker,
      //workers need to have enough RAM to fit this for efficient training
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)

      //This value is used for decaying the learning rate
      val numWordsProcessedInPreviousIterations = (k - 1) * trainWordsCount

      //Distributed sparkJob for performing iteration of training on each partition of
      //the training set
      val partial = newSentences.mapPartitionsWithIndex {
        case (idx, iter) =>
          //idx is the index of the partition and iter represents the sentences in this partition
          LogHolderW2V.log.info(s"Starting Iteration ${k} on partition ${idx}")

          //Initialize local loss for this partition and this iteration
          var loss = 0.0f

          //PRNG for random initialization of vectors
          val random = Random

          //Store modifications made to the word embeddings and the hidden weights
          //These values are local to each executor and are updated during training,
          //Synchronization with the driver happens only between each iteration.
          val syn0Modify = new Array[Int](vocabSize)
          val syn1Modify = new Array[Int](vocabSize)

          //Fold over all sentences of vocab indexes in this partition
          //Initialize accumulators with the word embeddings, the hidden weights, and zeros
          //where the zeros represents the lastWordCount and wordCount.
          val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
            case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
              var lwc = lastWordCount
              var wc = wordCount

              //Logging and decay learning rate every 10000 words
              if (wordCount - lastWordCount > 10000) {
                lwc = wordCount
                //Decay learning rate
                alpha = learningRate *
                  (1 - (numPartitions * wordCount.toDouble + numWordsProcessedInPreviousIterations) /
                    totalWordsCounts)

                //Lower bound of learning rate
                if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001

                //Logging
                if (verbose) {
                  LogHolderW2V.log.info(s"wordCount = ${wordCount + numWordsProcessedInPreviousIterations}, " + s"alpha1 = $alpha")
                  LogHolderW2V.log.info(s"avg loss: ${(loss + globalLoss) / (numWordsProcessedInPreviousIterations + wordCount)}")
                }
              }
              //Increment wordcount with the number of words in this sentence
              wc += sentence.length
              //Keep track of position in the sentence
              var pos = 0
              while (pos < sentence.length) {
                //Obtain word at the given position in sentence
                val word = sentence(pos)
                //Sample context window size [0, window]
                val b = random.nextInt(window)
                // Train Skip-gram
                var a = b
                //Loop over all context words
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = pos - window + a //Get position of context word
                    if (c >= 0 && c < sentence.length) {
                      val lastWord = sentence(c) //Get context word
                      //Calculate index in the long flattened vector of the specific context word
                      val l1 = lastWord * vectorSize
                      //Placeholder for error accumulation for the predicted word embedding
                      val neu1e = new Array[Float](vectorSize)
                      // Hierarchical softmax, O(Log N) tree traversal
                      //During the traversal, predict the path to take (left or right)
                      //The probability of a given leaf-node (a word) is the joint probability of
                      //taking the right path (making correct decision at each node)
                      var d = 0
                      while (d < bcVocab.value(word).codeLen) {
                        //Get the path of nodes from root to center word and select the node "d"
                        //Note that this prediction is "reversed" since we predict the context words
                        //conditioned on the center word (it is equivalent)
                        val inner = bcVocab.value(word).point(d)
                        //Get index of the current node in the traversal.
                        //The index is used to access the vector representations of the word in the long flattened vectors
                        val l2 = inner * vectorSize
                        // Propagate hidden -> output

                        //f is the dot product between word embedding syn0 of the context word
                        //and the weight from the hidden layer (syn1) to the intermediate node
                        //in the tree traversal.
                        //blas.sdot(Number of elements in input vector, x-array, x-offset,
                        //increment-x (storage spacing between elements of x, y-array, y-offset,
                        //increment-y (storage spacing between elements of y)
                        var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)

                        //if the EXP value of f exists in the precomputed table, use it
                        if (f > -MAX_EXP && f < MAX_EXP) {
                          //Obtain the "range" or "piece" in the lookup table corresponding to f
                          val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                          //Obtain exp(f) through the lookup table
                          f = expTable.value(ind)
                          //Binary logistic regression to predict the correct path in the traversal
                          //bcVocab.value(word).code(d) is the label
                          //f is the prediction
                          //bcVocab.value(word).code(d) - f is the derivative/gradient of cross-entropy loss
                          //Multiply gradient of softmax with learning rate
                          //To get loss, compute -log(1 - f) if label is 0, otherwise -log(f)
                          //1 - gradientExpression due to the reverse prediction.
                          val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat

                          //Compute loss for logging purposes
                          if (verbose) {
                            if (bcVocab.value(word).code(d) == 0) {
                              if (f > 1.0)
                                loss += 0.0f
                              else
                                loss += -(Math.log(f.toDouble).toFloat)
                            } else {
                              if ((1.0f - f) > 1.0)
                                loss += 0.0f
                              else
                                loss += -(Math.log((1.0f - f).toDouble).toFloat)
                            }
                          }
                          //Propagate errors from output to hidden (layer closest to error)
                          //Store errors in neu1e.
                          //saxpy computes z = alpha * x + y,
                          //i.e a constant times a vector plus another vector
                          //saxpy(number of elements in vector, constant, x-vector, x-offset,
                          //increment-x (storage spacing between elements of x, y-vector, y-offset,
                          //increment-y (storage spacing between elements of y)
                          //This computation takes g*syn1 + neu1e and stores the result in neu1e
                          //syn1 is the hidden weight to the node in the tree, g is the gradient
                          //and neu1e is a vector of deltas for each position in the vector.
                          //neu1e is accumulated through the traversal and will be used to update
                          //the word embedding of the leaf node
                          blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                          //This line computes g*syn0 + syn1 and stores the result in syn1
                          //I.e this line updates the vector representation of the current node in
                          //the tree.
                          blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                          //Set flag that the vector of this node was modified
                          syn1Modify(inner) += 1
                        }
                        d += 1
                      }
                      //Learn the word embeddings, update the word embedding based on the accumulated
                      //deltas of neu1e. This computation takes 1*neu1e + syn0 and stores in syn0
                      //on the index for the given word.
                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                      //Set flag that the embedding of this word was updated
                      syn0Modify(lastWord) += 1
                    }
                  }
                  //Move to next context word in window
                  a += 1
                }
                //Move to next center word in sentence
                pos += 1
              } //Sentence completed
              //Updated accumulators for embeddings and hidden weights as well as word counts
              (syn0, syn1, lwc, wc)
          } //Partition completed for this iteration

          //Extract the newly computed embeddings and weights based on this partition
          //These updates should be sent to the driver to update the global values
          val syn0Local = model._1
          val syn1Local = model._2
          // Only output modified vectors.
          // Extract all modified values from syn0 and syn1 by looking at the flags in syn0Modify and syn1Modify
          // The extracted values is the return value for the training on this partition of the dataset.
          //The form of the output is list of (index, vector)
          Iterator.tabulate(vocabSize) { index =>
            if (syn0Modify(index) > 0) {
              Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
            } else {
              None
            }
          }.flatten ++ Iterator.tabulate(vocabSize + 1) { index =>
            if (index == vocabSize) {
              Some((-1, Array[Float](loss)))
            } else {
              if (syn1Modify(index) > 0) {
                Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
              } else {
                None
              }
            }
          }.flatten //Return value of this partition and this iteration, i.e deltas of all updated vectors are sent to driver
      }

      var synAgg = Array[(Int, Array[Float])]()
      if (average) {
        //Synchronize the partial syn0/syn1 updates from all partitions/executors
        //Synchronize by averaging over all the deltas for same word/node
        synAgg = partial.groupByKey().map(e => {
          val count = e._2.size
          val g: Iterable[Array[Float]] = e._2
          val sumUpd = g.foldLeft(new Array[Float](vectorSize))((acc, vec) => {
            blas.saxpy(vec.size, 1.0f, vec, 1, acc, 1)
            acc
          })
          val scalar = 1.0f / (count.toFloat)
          blas.sscal(sumUpd.size, scalar, sumUpd, 1)
          (e._1, sumUpd)
        }).collect()
      } else {
        //Synchronize the partial syn0/syn1 updates from all partitions/executors
        //Synchronize by summing over all the deltas for same word/node
        synAgg = partial.reduceByKey {
          case (v1, v2) =>
            blas.saxpy(v2.size, 1.0f, v2, 1, v1, 1)
            v1
        }.collect()
      }

      //Extract the summed loss over all partitions for this iteration
      val incrLoss = synAgg.filter((e) => e._1 == -1)(0)._2(0)
      //Increment the global loss
      globalLoss += incrLoss
      LogHolderW2V.log.info(s"avg loss: ${globalLoss / (numWordsProcessedInPreviousIterations + trainWordsCount)}")
      //Extract the updated weigths
      val synAgg1 = synAgg.filter((e) => e._1 != -1)

      //Update the global values of hidden weights and embeddings with the learned values of this iteration
      var i = 0
      while (i < synAgg1.length) {
        val index = synAgg1(i)._1
        if (index < vocabSize) {
          //Copy synAgg(i) to syn0Global
          Array.copy(synAgg1(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        } else {
          //Copy synAgg(i) to syn1Global
          Array.copy(synAgg1(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      //Destroy the read-only variables of the hidden and embeddings from the previous iteration as those are now outdated
      bcSyn0Global.destroy()
      bcSyn1Global.destroy()
    } //Next iteration

    //All iterations are finnished and embeddings are learned, no need to persist the dataset of raw words in memory any more.
    newSentences.unpersist()
    //Get array of all words
    val wordArray = vocab.map(_.word)
    val wordIndex = wordArray.zipWithIndex.toMap
    val wordVectors = syn0Global
    new Word2VecModel(wordIndex, wordVectors)
  }

  /**
   * Resulting Model from Word2Vec training
   */
  class Word2VecModel(val wordIndex: Map[String, Int], val rawWordVectors: Array[Float]) {

    val wordVectors = convertRawVectorsToWordVectors(rawWordVectors)

    /**
     * Returns a map of words to their vector representations.
     * Normalized or un-normalized.
     *
     * @param rawWordVectors word vectors in a long array that have not been normalized
     * @return map of (String, WordVector)
     */
    def convertRawVectorsToWordVectors(rawWordVectors: Array[Float]): Map[String, Array[Float]] = {
      if (norm) {
        val norms = wordVecNorms(rawWordVectors)
        val normalized = normalizeVecs(rawWordVectors, norms)
        wordIndex.map {
          case (word, ind) =>
            (word, normalized.slice(vectorSize * ind, vectorSize * ind + vectorSize))
        }
      } else {
        wordIndex.map {
          case (word, ind) =>
            (word, rawWordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
        }
      }
    }

    /**
     * Save output to Word2Vec textual format for interoperability with for example gensim
     *
     * @param wordVectors map of words and their corresponding vectors
     * @param outputPath the path to save the vectors
     * @param sc the spark context
     * @param parallel boolean flag indicating whether to save in parallelized format
     */
    def saveToWord2VecFormat(wordVectors: Map[String, Array[Float]], outputPath: String, sc: SparkContext, parallel: Boolean): Unit = {
      val vocabSize = wordVectors.size
      val dim = wordVectors.head._2.size
      val header = vocabSize + " " + dim
      val temp = wordVectors.toList.map((x) => x._1 + " " + x._2.map(v => "%f".formatLocal(java.util.Locale.US, v)).mkString(" "))
      val w2vStringFormat = header :: temp
      if (parallel)
        sc.parallelize(w2vStringFormat).saveAsTextFile(outputPath)
      else {
        val writer = new BufferedWriter(new FileWriter(outputPath))
        w2vStringFormat.foreach((s) => writer.write(s + "\n"))
        writer.close()
      }
    }

    /**
     * Computes the L2 norm of each vector
     *
     * @param rawWordVectors the wordvectors represented by a long array
     * @return array with L2Norm for each word vector at its index
     */
    def wordVecNorms(rawWordVectors: Array[Float]): Array[Float] = {
      val wordVecNorms = new Array[Float](vocabSize)
      var i = 0
      while (i < vocabSize) {
        val vec = rawWordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
        wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
        i += 1
      }
      wordVecNorms
    }

    /**
     * Normalize vectors with the L2 norm
     *
     * @param rawWordVectors the wordvectors represneted by a long array
     * @param wordVecNorms an array with L2Norms for the word vectors
     * @return array with normalized word vectors
     */
    def normalizeVecs(rawWordVectors: Array[Float], wordVecNorms: Array[Float]): Array[Float] = {
      var i = 0
      while (i < vocabSize) {
        val l2scalar = (1.0f / wordVecNorms(i)).toFloat
        blas.sscal(vectorSize, l2scalar, rawWordVectors, i * vectorSize, 1)
        i += 1
      }
      rawWordVectors
    }

  }
}

/**
 * Utility to make sure logger is serializable
 */
object LogHolderW2V extends Serializable {
  @transient lazy val log = LogManager.getRootLogger()
  log.setLevel(Level.INFO)
}
