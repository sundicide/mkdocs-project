# fdd-in-scala

## Week1-1: Recap: Functions and Pattern Matching

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

### Partial Function
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

<hr />

## Week1-2: Recap: Collections

All collection types share a common set of general methods.

Core Methods:
- map
- flatMap
- filter

and also
- foldLeft
- foldRight

```scala
abstract class List[+T] {
  def map[U](f: T => U): List[U] = this match {
    case x :: xs => f(x) :: xs.map(f)
    case Nil => Nil
  }
}
```

```scala
abstract class List[+T] {
  def flatMap[U](f: T => List[U]): List[U] = this match {
    case x :: xs => f(x) ++ xs.flatMap(f)
    case Nil => Nil
  }
}
```
map과의 다른점
1. map과 달리 f가 List[U]를 리턴한다.
2. ++로 List를 concat한다.(List끼리의 concat이므로)

<hr />

## Week3-1: Type-Directed Programming
지금까지 봤듯이 compiler는 values로 부터 types을 유추할 수 있다.

```scala
val x = 12
```
compiler는 x를 Int로 유추한다. 왜냐하면 값이 12이므로
아래와 같이 복잡한 표현에서도 이는 적용된다.

```scala
val y = x + 3
```
compiler는 y 또한 Int로 유추한다.

<br />
이번에는 반대로 compiler가 types로 부터 values를 유추하는 과정을 볼 것이다.
왜 이것이 유용하냐? 확실한 하나는 compiler가 value를 찾아서 줄 수 있기 때문이다.

이번 레슨의 나머지는 이런 메카니즘의 motivation을 소개하고 다음 번 레슨은 how to use it을 설명할 것이다.

### Motivating Example
parameter로 List[Int]를 받아서 정렬한 결과를 List[Int]로 리턴하는 함수를 생각해보자.

```scala
def sort(xs: List[Int]): List[Int] = {
  ...
  ... if (x < y) ...
  ...
}
```
상세 코드는 여기에서 필요가 없기에 생략했다. 위 코드는 Int에 대해서만 적용 가능하므로 general하게 모든 타입에 대해서도 동작하게 하고 싶다.
이에 대한 straightforward approach는 polymorphic type을 사용하는 것이다.

```scala
def sort[A](xs: List[A]): List[A] = ...
```

하지만 이것만으로는 부족하다. 왜냐하면 각 type별로 compare를 다르게 해야 하기 때문이다.
그래서 이번엔 각 compare 함수를 parameter로 받도록 해보자.

```scala
def sort[A](xs: List[A])(lessThan: (A, A) => Boolean): List[A] = {
  ...
  ... if (lessThan(x, y)) ...
  ...
}
```

그렇게 되면 아래와 같이 가능하다'

```scala
val xs = List(-5, 6, 3, 2, 7)

val strings = List("apple", "pear", "orange", "pineapple")

sort(xs)((x, y) => x < y)

sort(strings)((s1, s2) => s1.compareTo(s2) < 0)
```

### Refactoring With Ordering
scala는 standard library 에서 comparing 하는 함수를 기본으로 제공한다.

```scala
package scala.math

trait Ordering[A] {
  def compare(a1: A, a2: A): Int
  def lt(a1: A, a2: A): Boolean = compare(a1, a2) <= 0
  ...
}
```
compare 함수는 2개의 parameter를 받아서 첫 번째 값이 클 경우 양수, 작을 경우 음수, 동일한 경우 0을 리턴한다.

이를 사용하면 아래와 같이 변경 가능하다.
```scala
def sort[A](xs: List[A])(ord: Ordering[A]): List[A] = {
  ...
  ... if (ord.lt(x, y)) ...
  ...
}
```
```scala
import scala.math.Ordering

sort(xs)(Ordering.Int)
sort(strings)(Ordering.String)
```

여기에서 사용 중인 Int와 String은 `types`이 아니고 `values`임을 알아야 한다.
scala에서는 types과 values에 동일한 symbol을 사용하는 것이 가능하다.

```scala
object Ordering {
  val Int = new Ordering[Int] {
    def compare(x: Int, y: Int) = if (x > y) 1 else if (x < y) -1 else 0
  }
}
```

### Reducing Boilerplate
지금까지 정의한 것을 따르면 잘 동작한다.
하지만 모든 경우에 대해 boilerplate가 존재하게 된다. Int를 비교할 때마다 `Ordering.Int`를 반복적으로 사용해야 한다.

```scala
sort(xs)(Ordering.Int)
sort(ys)(Ordering.Int)
sort(strings)(Ordering.String)
```

### Implicit Parameters

`implicit`을 명시함으로서 compiler가 argument `ord`를 support를 하게 할 수 있다.
```scala
def sort[A](xs: List[A])(implicit ord: Ordering[A]): List[A] = ...
```

```scala
sort(xs)
sort(ys)
sort(strings)
```

위와 같이 하면 컴파일러가 value에 맞춰 type을 결정한다.

컴파일러가 수행하는 과정을 자세히 살펴보자.
```scala
sort(xs)
```

xs 가 List[Int] 타입이므로 컴파일러는 위의 코드를 아래와 같이 변환한다.
```scala
sort[Int](xs)
```

그리고 컴파일러는 candidate definition중 Ordering[Int] 타입에 맞는 것을 찾는다. 위의 케이스에서는 Ordering.Int와 only matching되고 컴파일러는 method sort로 이를 전달한다.

```scala
sort[Int](xs)(Ordering.Int)
```

candidate values가 어떻게 정의도어있는 지를 살펴 보기 전에 implicit 키워드에 대해 자세히 알아보자.

1. method는 오직 하나의 implicit parameter list를 가질 수 있으며 이는 마지막 paramter가 되야 한다.
1. At call site, the arguments of the given clause are usually left out, although it is possible to explicitly pass them:
```scala
// Argument inferred by the compiler
sort(xs)

// Explicit argument
sort(xs)(Ordering.Int.reverse)
```

### Candidates for Implicit Parameters
컴파일러가 type T에 대해 어떤 candidate definition를 찾을까?
컴파일러는 아래 definition을 찾는다.

- have type T,
- are marked implicit,
- are visible at the point of the function call, or are defined in a companion object associated with T.

most specific한 정의를 찾게 되면 그것을 사용하고 없다면 error를 report한다.

#### implicit Definition
implicit definition이란 implicit 키워드와 함께 정의된 것을 말한다.

```scala
object Ordering {
  implicit val Int: Ordering[Int] = ...
}
```
위의 코드는 Int라는 이름을 가진 Ordering[Int] 타입의 implicit value를 정의한 것이다.

Any val, lazy val, def, or object definition can be marked implicit.

마지막으로 implicit definitions는 type parameters와 implicit parameters를 가질 수 있다.

```scala
implicit def orderingPair[A, B](implicit
  orderingA: Ordering[A],
  orderingB: Ordering[B]
): Ordering[(A, B)] = ...
```

#### Implicit Search Scope
type T의 implicit value를 찾기 위해 첫 번째로 visible(inherited, imported, or defined in an enclosing scope)한 모든 implicit definitions를 찾는다.

만약 컴파일러가 lexcial scope에서 implicit instance와 매칭되는 type T를 찾지 못하면, T와 관련된 companion objects에서 이어서 찾는다. (companion objects와 types는 other types와 연관있다.)

A companion object is an object that has the same name as a type. 예로 object scala.math.Ordering is the companion of the type scala.math.Ordering.

The types associated with a type T are:

- if T has parent types T₁ with T₂ ... with Tₙ, the union of the parts of T₁, ... Tₙ as well as T itself,
- if T is a parameterized type S[T₁, T₂, ..., Tₙ], the union of the parts of S and T₁, ..., Tₙ,
- otherwise, just T itself.

As an example, consider the following type hierarchy:

```scala
trait Foo[A]
trait Bar[A] extends Foo[A]
trait Baz[A] extends Bar[A]
trait X
trait Y extends X
```

만약 Bar[Y] 타입의 implicit value가 필요하다면 compiler는 아래와 같은 companion object에서 implicit definition을 찾을 것이다.

- Bar, because it is a part of Bar[Y],
- Y, because it is a part of Bar[Y],
- Foo, because it is a parent type of Bar,
- and X, because it is a parent type of Y.
- However, the Baz companion object will not be visited.

#### Implicit Search Process
search process는 no candidate found 혹은 매칭되는 최소한 하나의 candidate를 결과를 만들어 낸다.

만약 no no available implicit definition matching 이라면 에러가 repot 된다.

```scala
scala> def f(implicit n: Int) = ()

scala> f
       ^
error: could not find implicit value for parameter n: Int
```

반대로 둘 이상의 implicit definition이 eligibale 하다면 ambiguity가 report 된다.

```scala
scala> implicit val x: Int = 0

scala> implicit val y: Int = 1

scala> def f(implicit n: Int) = ()

scala> f
       ^
error: ambiguous implicit values:
  both value x of type => Int
```

same type에 매칭되는 several implicit definitions가 있어도 하나를 특정 할 수 있다면 괜찮다.

A definition a: A is more specific than a definition b: B if:

- type A has more “fixed” parts,
- or, a is defined in a class or object which is a subclass of the class defining b.

Let’s see a few examples of priorities at work.

Which implicit definition matches the Int implicit parameter when the following method f is called?

```scala
implicit def universal[A]: A = ???
implicit def int: Int = ???
def f(implicit n: Int) = ()
f
```

위의 경우에서 universal은 type paramter를 지니고 int는 아니기에, int가 more fixed parts를 갖고 이는 universal보다 먼저 고려된다. 그렇기 때문에 컴파일러가 int를 선택함에 있어 ambiguity가 없다.

아래와 같이 있을 때 implicit Int 파라미터를 갖는 f method는 어느 implicit definition에 매치 될까?

```scala
trait A {
  implicit val x: Int = 0
}
trait B extends A {
  implicit val y: Int = 1
  def f(implicit n: Int) = ()
  f
}
```

y가 A를 extend하는 trait이므로 y가 A보다 more specific 하다. 그러므로 컴파일러가 y를 선택하는 것에 ambiguity는 없다.


### Context Bounds
Syntactic sugar allows the omission of the implicit parameter list:

```scala
def printSorted[A: Ordering](as: List[A]): Unit = {
  println(sort(as))
}
```

Type parameter A has one context bound: Ordering. This is equivalent to writing:

```scala
def printSorted[A](as: List[A])(implicit ev1: Ordering[A]): Unit = {
  println(sort(as))
}
```

More generally, a method definition such as:

```scala
def f[A: U₁ ... : Uₙ](ps): R = ...
```

Is expanded to:
```scala
def f[A](ps)(implicit ev₁: U₁[A], ..., evₙ: Uₙ[A]): R = ...
```


### Implicit Query
At any point in a program, one can query an implicit value of a given type by calling the implicitly operation:

```scala
scala> implicitly[Ordering[Int]]
res0: Ordering[Int] = scala.math.Ordering$Int$@73564ab0
```

Note that implicitly is not a special keyword, it is defined as a library operation:

```scala
def implicitly[A](implicit value: A): A = value
```


### Summary
In this lesson we have introduced the concept of type-directed programming, a language mechanism that infers values from types.

There has to be a unique (most specific) implicit definition matching the queried type for it to be selected by the compiler.

Implicit values are searched in the enclosing lexical scope (imports, parameters, inherited members) as well as in the implicit scope of the queried type.

The implicit scope of type is made of implicit values defined in companion objects of types associated with the queried type.

<hr />

## Week3-2: Type Classes
In the previous lectures we have seen a particular pattern of code combining parameterized types and implicits.
We have defined a parameterized type Ordering[A], implicit instances of that type for concrete types A, and implicit parameters of type Ordering[A]:

```scala
trait Ordering[A] {
  def compare(a1: A, a2: A): Int
}

object Ordering {
  implicit val Int: Ordering[Int] =
    new Ordering[Int] {
      def compare(x: Int, y: Int) = if (x < y) -1 else if (x > y) 1 else 0
    }
  implicit val String: Ordering[String] =
    new Ordering[String] {
      def compare(s: String, t: String) = s.compareTo(t)
  }
}

def sort[A: Ordering](xs: List[A]): List[A] = ...
```

We say that Ordering is a type class.

Type classes provide yet another form of polymorphism.
The method sort can be called with lists containing elements of any type A for which there is an implicit value of type Ordering[A].

At compile-time, the compiler resolves the specific Ordering implementation that matches the type of the list elements.


### Retroactive Extension
Type classes let us add new features to data types without changing the original definition of these data types. For instance, consider the following Rational type, modeling a rational number:

```scala
/** A rational number
  * @param num   Numerator
  * @param denom Denominator
  */
case class Rational(num: Int, denom: Int)
```

We can add the capability "to be compared" to the type Rational by defining an implicit instance of type Ordering[Rational]:

```scala
object RationalOrdering {
  implicit val orderingRational: Ordering[Rational] =
    new Ordering[Rational] {
      def compare(q: Rational, r: Rational): Int =
        q.num * r.denom - r.num * q.denom
    }
}
```

### Laws
So far, we have shown how to implement instances of a type class, for some specific types (Int, String, and Rational).

Now, let’s have a look at the other side: how to use (and reason about) type classes.

For example, the sort function is written in terms of the Ordering type class, whose implementation is itself defined by each specific instance, and is therefore unknown at the time the sort function is written. If an Ordering instance implementation is incorrect, then the sort function becomes incorrect too!

To prevent this from happening, type classes are often accompanied by laws, which describe properties that instances must satisfy, and that users of type classes can rely on.

Can you think of properties that instances of the type class Ordering must satisfy, so that we can be confident that the method sort won’t be broken?

Instances of the Ordering[A] type class must satisfy the following properties:

- inverse: the sign of the result of comparing x and y must be the inverse of the sign of the result of comparing y and x,
- transitive: if a value x is lower than y and that y is lower than z, then x must also be lower than z,
- consistent: if two values x and y are equal, then the sign of the result of comparing x and z should be the same as the sign of the result of comparing y and z.

The authors of a type class should think about such kind of laws and they should provide ways for instance implementers to check that these laws are satisfied.


### Example of Type Class: Ring

Let’s see how we can define a type class modeling a ring structure. A ring is an algebraic structure defined as follows according to Wikipedia:

In mathematics, a ring is one of the fundamental algebraic structures used in abstract algebra. It consists of a set equipped with two binary operations that generalize the arithmetic operations of addition and multiplication. Through this generalization, theorems from arithmetic are extended to non-numerical objects such as polynomials, series, matrices and functions.

This structure is so common that, by abstracting over the ring structure, developers could write programs that could then be applied to various domains (arithmetic, polynomials, series, matrices and functions).

A ring is a set equipped with two binary operations, + and *, satisfying the following laws (called the ring axioms):

(a + b) + c = a + (b + c)	+ is associative
a + b = b + a	+ is commutative
a + 0 = a	0 is the additive identity
a + -a = 0	-a is the additive inverse of a
(a * b) * c = a * (b * c)	* is associative
a * 1 = a	1 is the multiplicative identity
a * (b + c) = a * b + a * c	left distributivity
(b + c) * a = b * a + c * a	right distributivity
Here is how we can define a ring type class in Scala:

```scala
trait Ring[A] {
  def plus(x: A, y: A): A
  def mult(x: A, y: A): A
  def inverse(x: A): A
  def zero: A
  def one: A
}
```

Here is how we define an instance of Ring[Int]:

```scala
object Ring {
  implicit val ringInt: Ring[Int] = new Ring[Int] {
    def plus(x: Int, y: Int): Int = x + y
    def mult(x: Int, y: Int): Int = x * y
    def inverse(x: Int): Int = -x
    def zero: Int = 0
    def one: Int = 1
  }
}
```

Finally, this is how we would define a function that checks that the + associativity law is satisfied by a given Ring instance:

```scala
def plusAssociativity[A](x: A, y: A, z: A)(implicit ring: Ring[A]): Boolean =
  ring.plus(ring.plus(x, y), z) == ring.plus(x, ring.plus(y, z))
```

Note: in practice, the standard library already provides a type class Numeric, which models a ring structure.


### Summary
In this lesson we have identified a new programming pattern: type classes.

Type classes provide a form of polymorphism: they can be used to implement algorithms that can be applied to various types. The compiler selects the type class implementation for a specific type at compile-time.

A type class definition is a trait that takes type parameters and defines operations that apply to these types. Generally, a type class definition is accompanied by laws, checking that implementations of their operations are correct.

<hr />

## Week3-3: Conditional Implicit Definitions

In this lesson, we will see that implicit definitions can themselves take implicit parameters.

Let’s start with an example. Consider how we order two String values: is "abc" lexicographically before "abd"?

To answer this question, we need to compare all the characters of the String values, element-wise:

- is a before a? No.
- is b before b? No.
- is c before d? Yes!
- We conclude that "abc" is before "abd".

So, we compare two sequences of characters with an algorithm that compares the characters of the sequences element-wise. Said otherwise, we can define an ordering relation for sequence of characters based on the ordering relation for characters.

Can we generalize this process to sequences of any element type A for which there is an implicit Ordering[A] instance?

The signature of such an Ordering[List[A]] definition takes an implicit parameter of type Ordering[A]:

```scala
implicit def orderingList[A](implicit ord: Ordering[A]): Ordering[List[A]]
```

For reference, a complete implementation is shown below. You can see that at some point in the algorithm we call the operation compare of the ord parameter:

```scala
implicit def orderingList[A](implicit ord: Ordering[A]): Ordering[List[A]] =
  new Ordering[List[A]] {
    def compare(xs: List[A], ys: List[A]) =
      (xs, ys) match {
        case (x :: xsTail, y :: ysTail) =>
          val c = ord.compare(x, y)
          if (c != 0) c else compare(xsTail, ysTail)
        case (Nil, Nil) => 0
        case (_, Nil)   => 1
        case (Nil, _)   => -1
      }
  }
```

With this definition, we can sort a list of list of numbers, for example:

```scala
scala> val xss = List(List(1, 2, 3), List(1), List(1, 1, 3))
res0: List[List[Int]] = List(List(1, 2, 3), List(1), List(1, 1, 3))

scala> sort(xss)
res1: List[List[Int]] = List(List(1), List(1, 1, 3), List(1, 2, 3))
```

But let’s take a step back. We haven’t defined an instance of Ordering[List[Int]] and yet we have been able to sort a list of List[Int] elements! How did the compiler manage to provide such an instance to us?

This happened in several steps.

First, we called sort(xss). The compiler fixed the type parameter A of the method to List[Int], based on the type of the argument xss, as if we had written:

```scala
sort[List[Int]](xss)
```

Then, the compiler searched for an implicit definition of type Ordering[List[Int]]. It found that our orderingList definition could be a match under the condition that it could also find an implicit definition of type Ordering[Int], which it eventually found. Finally, the compiler inserted the following arguments for us:

```scala
sort[List[Int]](xss)(orderingList(Ordering.Int))
```

In this case, the compiler combined two implicit definitions (orderingList and Ordering.Int) before terminating. In general, though, an arbitrary number of implicit definitions can be combined until the search hits a “terminal” definition.

Consider for instance these four implicit definitions:

```scala
implicit def a: A = ...
implicit def aToB(implicit a: A): B = ...
implicit def bToC(implicit b: B): C = ...
implicit def cToD(implicit c: C): D = ...
```

We can then ask the compiler to summon a value of type D:

```scala
implicitly[D]
```

The compiler finds that there is a candidate definition, cToD, that can provide such a D value, under the condition that it can also find an implicit definition of type C. Again, it finds that there is a candidate definition, bToC, that can provide such a C value, under the condition that it can also find an implicit definition of type B. Once again, it finds that there is candidate definition, aToB, that can provide such a B value, under the condition that it can also find an implicit value of type A. Finally, it finds a candidate definition for type A and the algorithm terminates!

At the beginning of this lesson, we showed that by using implicit parameters the compiler could infer simple arguments for us. We have now reached a point where we can appreciate that the compiler can infer more complex arguments (by inferring arguments of arguments!).

It not only significantly reduces code verbosity, it also alleviates developers from implementing parts of their programs, which are summoned by the compiler based on their type (hence the name “type-directed programming”). In practice, complex fragments of programs such as serializers and deserializers of data types can be summoned by the compiler.


### Recursive Implicit Definitions
What happens if we write an implicit definition that depends on itself?

```scala
trait X

implicit def loop(implicit x: X): X = x

implicitly[X]
```

The compiler detects that it keeps searching for an implicit definition of the same type and returns an error:

```
error: diverging implicit expansion for type X
starting with method loop
```

Note: it is possible to write recursive implicit definitions by making sure that the search always terminates, but this is out of the scope of this lesson.


### Example: Sort by Multiple Criteria
Consider a situation where we want to compare several movies. Each movie has a title, a rating (in number of “stars”), and a duration (in minutes):

```scala
case class Movie(title: String, rating: Int, duration: Int)

val movies = Seq(
  Movie("Interstellar", 9, 169),
  Movie("Inglourious Basterds", 8, 140),
  Movie("Fight Club", 9, 139),
  Movie("Zodiac", 8, 157)
)
```

We want to sort movies by rating first, and then by duration.

To achieve this, a first step is to change our sort function to take as parameter the sort criteria in addition to the elements to sort:

```scala
def sort[A, B](elements: Seq[A])(critera: A => B)(implicit
  ord: Ordering[B]
): Seq[A] = ...
```

The sort algorithm remains the same except that instead of comparing the elements together, we compare the criteria applied to each element.

With this function, here is how we can sort movies by title:

```scala
sort(movies)(_.title)
```

And here is how we can sort them by rating:

```scala
sort(movies)(_.rating)
```

Each time the sort function is called, its ordering parameter is inferred by the compiler based on the type of the criteria (String and then Int, in the above examples).

However, our initial problem was to sort the movies by multiple criteria. We would like to sort first by rating and then by duration:

```scala
sort(movies)(movie => (movie.rating, movie.duration))
```

The type of the criteria is now a tuple type (Int, Int). Unfortunately, the compiler is unable to infer the corresponding ordering parameter. We need to define how simple orderings can be combined together to get an ordering for multiple criteria.

We do so by defining the following implicit ordering:

```scala
implicit def orderingPair[A, B](implicit
  orderingA: Ordering[A],
  orderingB: Ordering[B]
): Ordering[(A, B)] = ...
```

This definition provides an ordering for pairs of type (A, B) given orderings for types A and B.

The complete implementation is the following:

```scala
implicit def orderingPair[A, B](implicit
  orderingA: Ordering[A],
  orderingB: Ordering[B]
): Ordering[(A, B)] =
  new Ordering[(A, B)] {
    def compare(pair1: (A, B), pair2: (A, B)): Int = {
      val firstCriteria = orderingA.compare(pair1._1, pair2._1)
      if (firstCriteria != 0) firstCriteria
      else orderingB.compare(pair1._2, pair2._2)
    }
  }
```

We first compare the two values according to the first criteria, and if they are equal we compare them according to the second criteria.

With this definition, the compiler is now able to infer the ordering for the following call:

```scala
sort(movies)(movie => (movie.rating, movie.duration))
```

Here is the same call where the inferred parameter is explicitly written:

```scala
sort(movies)(movie => (movie.rating, movie.duration))(
  orderingPair(Ordering.Int, Ordering.Int)
)
```

Note that in the standard library the sort function that we have defined here is already available as a method sortBy on collections.


### Summary
In this lesson, we have seen that:

- implicit definitions can also take implicit parameters,
 an arbitrary number of implicit definitions can be chained until a terminal definition is reached.


## Wee5: Timely Effects

### Lecture 5.1 - Imperative Event Handling: The Observer Pattern
The Observer Pattern은 model의 변화에 따라 views가 변경되는 것이다.

- publish / subscribe
- mode/view/controller(MVC)
라고도 불린다.

```scala
trait Publisher {
  private var subscribers: Set[Subscriber] = Set()
  def subscribe(subscriber: Subscriber): Unit = subscribers += subscriber
  def unsubscribe(subscriber: Subscriber): Unit = subscribers -= subscriber
  def publish(): Unit = subscribers.foreach(_.handler(this))
}

trait Subscriber {
  def handler(pub: Publisher)
}

class BankAccount extends Publisher {
  private var balance = 0
  def currentBalance: Int = balance
  def deposit(amount: Int): Unit =
    if (amount > 0) {
      balance = balance + amount
      publish()
    }
  def withdraw(amount: Int): Unit =
    if (0 < amount && amount <= balance) {
      balance = balance - amount
      publish()
    } else throw new Error("insufficient funds")
}


class Consolidator(observed: List[BankAccount]) extends Subscriber {
  observed.foreach(_.subscribe(this))

  private var total: Int = _
  compute()

  private def compute() =
    total = observed.map(_.currentBalance).sum

  def handler(pub: Publisher) = compute()

  def totalBalance = total
}

val a = new BankAccount
val b = new BankAccount
val c = new Consolidator(List(a, b))

c.totalBalance
a deposit 20
c.totalBalance
b deposit 30
c.totalBalance
```
위 코드는 BankAccount를 Observer Pattern으로 구현한 것이다.

Observer Pattern의 장점은
- view를 state로부터 분리할 수 있다.
- 주어진 state로 여러 개의 views를 만들 수 있다.
- set up이 간단하다.

단점은
- handler가 Unit-typed여서 명령형 스타일이다.
- 많은 moving parts가 co-ordinated 되어야 한다.
- Concurrency가 문제를 복잡하게 만든다.
- View가 state에 강하게 bound되서 즉시 update가 일어난다. 가끔 view와 state간에 looser asynchronous relationship을 만들고 싶을때 단점으로 작용한다.


### Lecture 5.2 - Functional Reactive Programming
#### What is FRP
Reactive Programming은 in time에 일어난 이벤트들의 sequence에 reacting하는 것이다.

Functional view: event sequence를 signal로 합칠 수 있다.

- signal은 계속해서 변하는 value이다.
- mutable state를 계속 update 하는 대신 이미 있는 signal을 new로 정의할 수 있다.

#### Example: Mouse Positions
- Event-based view:
마우스가 움직일 때마다
```scala
MouseMoved(toPos: Position)
```
이 fired 된다.

- FRP view:
```scala
mousePosition: Signal[Position]
```
현재 마우스 위치를 표현하는 signal이 있다.


#### Fundamental Signal Operations
2개의 기본 operation이 있다.
1. 현재 signal의 value를 얻는 operation. 우리가 정의할 라이브러리에서는 `()`로 표현한다.
    mousePosition()
2. define a signal in terms of other signal. 우리가 정의할 라이브러리에서는 Singal 생성자로 표현한다.
    ```scala
    def inReactangle(LL: Position, UR: Position): Signal[Boolean] =
      Signal {
        val pos = mousePosition()
        LL <= pos && pos <= UR
      }
    ```

#### Constant Signals
항상 same value를 갖는 signal을 정의할 수 있다.
```scala
val sig = Signal(3) // the signal that is always 3
```

시간에 따라 변하는 signal을 어떻게 정의할 것인가?
- mousePosition같이 외부에 정의된 signal을 map으로 순회하는 방법이 있다.
- 또는 `Var`를 사용하는 방법이 있다.

Signal의 Value는 immutable하다.
하지만 우리는 변경될 수 있는 Signal의 subclass인 Var를 구현할 것이다.
Var는 value를 현재의 값으로 변경해주는 "update" operation을 갖는다.

```scala
val sig = Var(3)
sig.update(5)
```

scala에서 update는 assignment로 쓸 수 있다.
예를 들어 arr이라는 이름의 array가 있다고 할 때
```scala
arr(i) = 0
// 은 아래와 같이 변환된다.
arr.update(i, 0)
```
update method는 다음과 같다.
```scala
class Array[T] {
  def update(idx: Int, value: T): Unit
}
```

일반적으로 f() = E는 f.update(E)로 축약할 수 있다.
그러므로
`sig.update(5)`
는 아래와 같이 축약할 수 있다.
`sig() = 5`


Var Signal은 mutable variables처럼 보인다.
`sig()`는 현재 값을 얻는 것이며
`sig() = newValue`는 update이기 때문이다.

하지만 중요한 차이점이 있다.
Var는 future points in time에서 자동적으로 계산되는 값을 가질 수 있다.
또한 mutable variables의 exists 매커니즘이 없고 모든 updates를 직접 propagate해야 한다.

```
a = 2
b = 2 * a
a = a + 1
b = 2 * a // a 값이 변했지만 b값이 자동적으로 업데이트 되지 않기 때문에 다시 넣어줘야 한다.
```
```
a() = 2
b() = 2 * a()
a() = 3 // a가 3으로 업데이트 됐으므로 b는 6으로 자동 업데이트 된다.
```

아래와 같은 경우는 처리할 수 없다.
```scala
s() = s() + 1
```
이는 s가 항상 자기 자신보다 1커야 한다는 의미이므로 불가하다.

#### Lecture 5.3 - A Simple FRP Implementation
##### Thread-Local State
global state를 synchronization하려면 concurrent access 문제가 생긴다.
block을 사용하게 되면 느려지고 dealock의 위험성이 있다.
이를 해겷하기 위한 방법으로 global state대신 thread-local state를 둘 수 있다.

thread-local state란 각 thread가 variable의 copy본에 접근한다는 의미이다.
그래서 global variable을 사용하지만 thread사이엔 공유가 안된다.
이를 scala에서는 scala.util.DynamicVariable로 지원한다.