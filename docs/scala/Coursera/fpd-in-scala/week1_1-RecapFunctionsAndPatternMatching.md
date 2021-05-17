# Week1-1: Recap: Functions and Pattern Matching

Pattern Matching의 예

```scala
  def show(json: JSON): String = json match {
    case JSeq(elems) =>
      "[" + (elems map show mkString ", ") + "]"
    case JObj(bindings) =>
      val assocs = bindings map {
        case (key, value) => "\"" + key + "\": " + show(value)
      }
      "{" + (assocs mkString ", ") + "}"
    case JNum(num) => num.toString
    case JStr(str) => "\"" + str + "\""
    case JBool(b) => b.toString
    case JNull => "null"
  }
```

scala에서 모든 concrete type은 class or trait 이다.
function도 예외가 아니다.

JBinding => String
은
scala.Function1[JBinding, String]
의 축약형이다.

scala.Function1은 trait이고 JBinding과 String은 type arguments이다.

## Partial Function
another subtype of function, special type이다.
function과 동일하게 apply를 가지면서 `isDefinedAt`을 추가로 갖는다.

```scala
trait PartialFunction[-A, +R] extends Function1[-A, +R] {
  def apply(x: A): R
  def isDefinedAt(x: A): Boolean
}
```

```scala
val f1: String => String = { case "ping" => "pong"}
f1("ping") // pong
f1("abc") // MatchError!!!

val f: PartialFunction[String, String] = { case "ping" => "pong" }

f.isDefinedAt("ping") // true
f.isDefinedAt("pong") // false
```

만약 PartialFunction type이 기대된다면 Scala Compiler는 아래와 같이 확장한다.

```scala
{ case "ping" => "pong" }
```

as follows:

```scala
new PartialFunction[String, String] {
  def apply(x: String) = x match {
    case ”ping” => ”pong”
  }
  def isDefinedAt(x: String) = x match {
    case ”ping” => true
    case _ => false
  }
}
```

### Excercise1
```scala
val f: PartialFunction[List[Int], String] = {
  case Nil => ”one”
  case x :: y :: rest => ”two”
}

f.isDefinedAt(List(1, 2, 3)) // true
```

```scala
val g: PartialFunction[List[Int], String] = {
  case Nil => ”one”
  case x :: rest =>
    rest match {
      case Nil => ”two”
    }
}

g.isDefinedAt(List(1,2,3)) // ture
g(List(1,2,3)) // Match Error!!!
```

위에서 보듯이 `isDefinedAt`은 outmost matching block만 검증해준다. 그렇기 때문에 g에서는 true가 리턴 되는 것이다.
실제로 사용을 해보면 `case Nil` 밖에 case가 없기 때문에 에러가 발생한다.
