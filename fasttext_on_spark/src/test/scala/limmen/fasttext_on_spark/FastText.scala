package limmen.fasttext_on_spark

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import org.scalatest.PrivateMethodTester._
import org.scalatest._

class FastTextSuite extends FunSuite with Matchers {

  //Test case for the scenario mentioned in the paper
  test("computeSubWords") {
    val word = "where"
    val index = 0
    val vocabSize = 1
    val minn = 3
    val maxn = 3
    val bucket = 2000000
    val ft = new FastText().setMinn(minn).setMaxn(maxn).setBucket(bucket)
    val computeSubwords = PrivateMethod[Array[Int]]('computeSubwords)
    val hash = PrivateMethod[BigInt]('hash)
    val subwords = ft invokePrivate computeSubwords(word, vocabSize, index)
    val truthLabelsStr = Array("<where>", "<wh", "whe", "her", "ere", "re>")
    val truthLabelsHash = truthLabelsStr.map(s => {
      if (s.equals("<where>"))
        0
      else
        vocabSize + ((ft invokePrivate hash(s)) mod bucket).intValue
    })
    subwords.foreach(h => assert(truthLabelsHash.contains(h)))
  }

}
