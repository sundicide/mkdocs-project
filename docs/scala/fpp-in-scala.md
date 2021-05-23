# fpp-in-scala

## Week1

### Lecture 1.1 - Programming Paradigms

functional Programming은 paradigm이다. classical imperative paradimg(Java or C)과 약간 다른.
scala에서는 이 2개의 paradigm을 합칠 수도 있다. 이는 다른 언어에서의 migration을 쉽게 해준다.

In science, a `paradigm` describes distinct concepts or thought patterns in some scientific discipline.

Main Programming Paradigms:
- imperative programming
- functional programming
- logic programming

object-oriented programming도 paradigm이라고 하는 사람들도 있지만 자신의 생각으로는 위 3개의 교차점에 있다고 생각한다.

#### Imperative Programming
- modifying mutable variables
- using assignments
- and control structures such as if-then-else, loops, break, continue, return

Von Neumann computer의 sequence를 이해하는 것은 imperative program을 이해하는 most common informal way이다.

> Processor <------BUS ------> Memory


Problem: Scaling up. How can we avoid conceptualizing programs word by word?

high-level abstractions(collections, polynomials, geomtric shapes, strings, documents..)를 정의하는 테크닉이 필요하다.

Ideally: Develop theories of collections, shapes, strings, ...

#### What is a theory
A theory consist of
- one or more data types
- operations on these types
- laws that describe the relationships between values and operations

보통 theory는 `mutations`를 describe하지 않는다!

mutation: identity는 유지하면서 something을 change하는 것이다.

##### Theories without mutations
theory of polynomials

> (a*x + b) + (c*x + d) = (a+c)*x + (b+d)

theory of strings

> (a ++ b) ++ c = a ++ (b ++ c)


#### Consequences for Programming
mathematical theroies를 따르면서 high-level concepts 구현을 하려면 mutation은 없어야 한다.
- theroies do not admit it
- mutation은 theories의 useful laws를 destoy 할 수 있다.

그러므로
- concentrate on defining theories for operators expressed as functions
- avoid mutations
- have powerful ways to abstract and compose functions

start of function programming means avoid mutations

#### Functional Programming
- In a restricted sense, FP means programming without mutable variables, assignments, loops, and other imperative control structures
- In a wider sense, FP meas focusing on the functions
- In particular, functions can be valuses that are produced, consumed, and composed
- All this becomes easier in a functional language

#### Functional Programming Language
- In a restricted sense, a functional programming language is one which does not have mutable variables, assignments, or imperative control structures.
- In a wider sense, a functional programming language enables the construction of elegant programs that focus on functions.
- In particular, functions in a FP language are first-class citizens. This means
    - they can be defined anywhere, including inside other functions
    - like any other value, they can be passed as parameters to functions and returned as results
    - as for other values, there exists a set operators to compose functions

#### Some functional programming languages
In the restricted sense:
- Pure Lisp, XSLT, XPath, XQuery, FP
- Haskell (without I/O Monad or UnsafePerformIO)

In the wider sense:
- Lisp, Scheme, Racket, Clojure ▶ SML, Ocaml, F#
- Haskell (full language)
- Scala
- Smalltalk, Ruby (!)

#### Why Functional Programming?
Functional Programming is becoming increasingly popular because it offers the following benfits.
- simpler reasoning principles
- better modularity
- good for exploiting parallelism for multicore and cloud computing.


#### my summary
우리는 수학을 배우면서 mutable variables를 배운 적이 없다.

1 + 1 = 2이고 a + b = 3 이라면 그냥 3인 것이다.
오늘은 a + b = 3 이었는데 내일은 a + b = 4일 순 없었다.
(ax^2 + bx + c 는 여러 값이 될 수 있겠지만)

하지만 imperative programming에서는 자연스러운 개념이다.
int a = 1;
int b = 2;
a + b = 3;

a = 4;
a + b = 6;

왜 수학적인 원칙을 꺼내 들었냐?
module화 때문이다.

프로그램이 복잡해지면서 모듈화는 필수이다.
잘 된 모듈화란 무엇일까? 항상 동일한 결과를 리턴하는 모듈일 것이다.

map 함수를 생각해보면 어느 타입에 상관없이
List[U]를 리턴한다.

이런 수학적 원칙들은 mutable variables를 인정하지 않는다. 그렇기에 functional programming language에 잘 맞는다.

fp는 이런 모호함을 제거함으로서 원칙을 보다 잘 구현하고 모듈화 하기 좋으며 multicore와 cloud computing 환경에서 병렬처리를 잘 할 수 있게 해준다.

<hr />

## substitution model
- 함수의 argument를 왼쪽부터 모두 평가
- 함수의 오른쪽부터 교체

-> 모든 Expression에 사용 가능 , No side effect

foundation of functional programming인 람다 calculus에 formalized 되어있다.

모든 Expr이 reduce to a value? (X)

아래와 같은 예가 있다.
```scala
def loop: Int = loop
```

### Evaluation Stratigies
CBV(Call By Value), CBN(Call By Name)
- CBV: 모든 args는 한 번만 평가한다는 장점
- CBN: 호출 될 때까지 not evaluted된다는 장점

만약 CBV가 종료된다면 CBN도 종료된다? (O)
반대로 CBN이 종료된다면 CBV도 종료된다? (X)

scala에서 CBN을 쓰는 방법은 parameter에 `=>`를 붙이면 된다.

```scala
def myFunc(a:=> Int) = a
```

### Value Definitions
```scala
val x = 2
val y = square(x) // 바로 평가된다.


def loop: Boolean = loop

def x = loop // (O) def는 호출될때 평가된다.
val x = loop // (X) Error

def and(x: Boolean, y: Boolean) =
  if (x) y else false

and(false, loop) // (X) Error

def and2(x: Boolean, y:=> Boolean) =
  if (x) y else false
and2(false, loop) // false
```

### Nested Functions
small func로 분리하는 것. good FP styles
sqrtIter, imporve 같은 함수들은 외부에 공개(direct 호출) 하고 싶지 않을 수 있다.

이러한 보조 함수들을 내부 함수로 둠으로써 name-space pollution을 방지할 수 있다.

```scala
def sqrt(x: Double) = {
  def improve
  def sqrtIter
}
```

### Lexical Scoping
outer block에 있는 definitions는 inside block에서 visible하다.

보통 문장 라인 끝 `;`는 optional이다.
다만 한 문장에 여러 expr을 표현할 때는 필수 이다.

```scala
val y = x + 1; y + y
```

### Tail Recursion
calls itself as its last action.
the function's stack frame can be reused
(one stack frame이 필요하며, tail calls 라고 함)

`@tailrec` annotation을 함수 위에 추가하면 해당 함수가 tail recur 하지 않을 시 오류가 발생한다.

아래의 factorial 함수는 tailrc 함수가 아니며 gcd는 tailrec 함수이다.

```scala
def gcd(a: Int, b: Int): Int =
  if (b == 0) a else gcd(b, a % b)

def factorial(n: Int): Int =
  if (n == 0) 1 else n * factorial(n - 1)
```

그 차이는 gcd는 스텝을 진행을 계속 하더라도 본인 호출만 계속 하게 되지만 factorial 같은 경우에는 좌측이 계속 늘어난다
> 4 * factorial(3)

이를 tail recursive 하게 변경하면 아래와 같다.

```scala
def factorial(n: Int): Int = {
  @tailrec
  def loop(acc: Int, n: Int): Int =
    if (n == 0) acc
    else loop(acc * n, n -1)
  loop(1, n)
}
```

Donal Knuth said premature optimization is the source of the evil

## Week2

### Higher order functions
pass functions as arguments and retun them as results.

functional languages treat functions as first-class values.
= like any other value, a function can be passed as a parameter and returned as a result.

provides a flexible way to compose program.



#### Anonymous Function
함수를 parameter로 전달하다보면 many small function을 만들게 된다. 그렇게 되면 각각의 naming을 정하는 것은 어렵게 된다.
=> anonymous function을 사용한다.

```scala
def str = "abc"; println(str)

println("abc")
```
위에는 str 변수를 정의해서 호출했고 아래는 정의 없이 사용 했다.
이것이 가능한 이유는 뭘까?
=> str은 literals 이기 때문이다.

마찬가지로 이름 없이 함수를 쓰면 function literals가 된다.
= anonymous functions

```scala
// cube anonymous func
(x: Int) => x * x * x
```
`(x: Int)`는 parameter
`x * x  * x`는 body


```scala
def sum(f: Int => Int, a: Int, b: Int): Int = {
  @tailrec
  def loop(a: Int, acc: Int): Int = {
    if (a > b) acc
    else loop(a + 1, f(a) + acc)
  }
  loop(a, 0)
}
def sumInts(a: Int, b: Int) = sum(x => x, a, b)

def sumCubes(a: Int, b: Int) = sum(x => x * x * x, a, b)
```

### Currying

아래 함수를 더 짧게 할 수는 없을까?
```scala
def sumInts(a: Int, b: Int) = sum(x => x, a, b)
```

```scala
def sum()
```

```scala
def sum(f: Int => Int)(a: Int, b: Int): Int =
  if (a > b) 0 else f(a) + sum(f)(a + 1, b)

def product(f: Int => Int)(a: Int, b: Int): Int =
  if (a > b) 1
  else f(a) * product(f)(a + 1, b)
```

위와 같은 스타일의 definition과 function을 currying이라고 부른다. Haskell Brooks Curry의 이름을 딴 네이밍이다.
Idea는 그 보다 전인 Schonfinkel과 Frege에 의해서 나왔지만 currying이란 네임으로 굳어졌다.


### Example: Finding Fixed Points
A number x is called a fixed point(고정 점) of a function f if

> f(x) = x

예로 f: x => 1 + x/2 라 할 때 fixed point는 2이다.
f(2) = 2이므로

몇몇 함수들은 f를 반복적으로 수행함으로서 fixed point를 찾을 수 있다.

> x, f(x), f(f(x)), f(f(f(x))), ...

initial estimate로 시작해서 f를 반복적으로 수행하다보면 더 이상 변하지 않는 값 혹은 변경이 충분히 적어졌을 때의 값을 fixed point라 부를 수 있다.

```scala
import math.abs

val tolerance = 0.0001
def isCloseEnough(x: Double, y: Double): Boolean =
  abs((x - y) / x) / x < tolerance

def fixedPoint(f: Double => Double)(firstGuess: Double) = {
  @tailrec
  def iterate(guess: Double): Double = {
    val next = f(guess)
    if (isCloseEnough(guess, next)) next
    else iterate(next)
  }
  iterate(firstGuess)
}
fixedPoint(x => 1 + x/2)(1) // 1.9975

def sqrt(x: Double) = fixedPoint(y => x / y)(1)
sqrt(2) // 무한 loop
```
위의 예에서 sqrt(2)를 수행하면 무한 loop가 발생한다.
1과 2 를 계속 반복한다
이를 해결 하기 위해서는 첫 번째 계산 값과 두 번째 계산 값의 평균을 구하면 된다.

```scala
def sqrt(x: Double) =
  fixedPoint(y => (y + x / y) / 2)(1)
```

#### functions as return values
위의 예제에서 평균을 통해 안정화시키는 기술은 추상화 될 수 있다.
```scala
def averageDamp(f: Double => Double)(x: Double) =
  (x + f(x)) / 2
def sqrt3(x: Double) =
  fixedPoint(averageDamp(y => x / y))(1)
```

Higher Order Function이 항상 옳은 것은 아니며 적절 할 때 사용해야 한다.


### Functions and Data
#### Classes
```scala
class Rational(x: Int, y: Int):
  def numer = x
  def denom = y
```
위 정의는 two entities를 생성한다.
- Rational 이라는 이름의 new type
- 이 type의 element를 만들기 위한 Rational constructo

스칼라는 types과 value의 names를 `different namespace`에 보관하기 때문에 충돌을 걱정할 필요 없다.

#### Objects
elements of a class type을 objects라고 부른다.
class 의 생성자를 calling 함으로서 object를 만들 수 있다.

```scala
Rational(1, 2)
```

아래와 같이 class내 member에 접근 가능하다

```scala
val x = Rational(1 ,2)
x.numer
x.denom
```

```scala
object rationals {
  val x = new Rational(1, 3)
  val y = new Rational(5, 7)
  val z = new Rational(3, 2)

  x.add(y).mul(z)
}
class Rational(x: Int, y: Int) {
  def numer = x
  def denom = y

  def add(r: Rational) =
    new Rational(numer * r.denom + r.numer * denom,
      denom * r.denom)

  def mul(r: Rational) =
    new Rational(numer * r.numer,
      denom * r.denom)

  def neg = new Rational(-numer, denom)

  def sub(r: Rational) = add(r.neg)

  override def toString = s"$numer/$denom"
}

```

### More Fun With Rationals
Client's view에서는 내부가 어떻게 동작하던지 동일하게 보인다.

without affecting client를 하면서 다른 구현을 선택하는 것을 `data abstraction`이라고 한다.
S/E에서의 cornerstone이다.

#### Self Reference
inside of a class, `this`는 현재 실행 중인 method내에서의 object를 의미한다

#### Preconditions
`require`로 class에 조건을 추가할 수 있다.
조건에 맞지 않으면 IllegalArgumentException이 발생하며 추가한 에러 메세지가 출력된다.

#### Assertions
require와 비슷한 의미이다.
require와 동일하게 condtion과 optional message string을 받는다.

```scala
val x = sqrt(y)
assert(x >= 0)
```

fail일 경우 assert는 require와 달리 AssertionError를 발생한다.

- require는 함수 호출자에게 precondition을 강요할 때 쓰인다
- assert는 함수 자신이 체크 할 때 사용한다.

#### Constructors
모든 class는 primary constructor(기본 생성자)가 암시적으로 있다.
- class의 모든 paramters를 받고
- class body의 모든 statement를 실행한다.

Java 같이 여러 생성자를 갖는 것도 가능하다.

```scala
object rationals {
  val x = new Rational(1, 3)
  val y = new Rational(5, 7)
  val z = new Rational(3, 2)

  x.add(y).mul(z)
  y.add(y)
  x.less(y)
  x.max(y)
  new Rational(2)
}
class Rational(x: Int, y: Int) {
  require(y != 0, "denominator must be nonezero")

  def this(x: Int) = this(x, 1) // 여기에서의 this는 constructor 의미로 쓰인다.

  private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
  private val g = gcd(x, y) // val로 선언했기에 바로 평가되서 다음 부턴 계산을 안하고 재사용한다.
  def numer = x / g
//    def numer = x / gcd(x,y) // 만약 이와 같이 선언 하면 매번 gcd를 계산해야 한다. 계산 리소스가 크고 가끔 호출될 때 사용하면 좋다.
  def denom = y / g

  def less(that: Rational) = numer * that.denom < that.numer * denom
  def max(that: Rational) = if (this.less(that)) that else this

  def add(r: Rational) =
    new Rational(numer * r.denom + r.numer * denom,
      denom * r.denom)

  def mul(r: Rational) =
    new Rational(numer * r.numer,
      denom * r.denom)

  def neg = new Rational(-numer, denom)

  def sub(r: Rational) = add(r.neg)

  override def toString = s"$numer/$denom"
}

```

<hr />

### Evaluation and Operators
#### Operators
##### Infix Notation
parameter를 갖는 모든 메소드는 infix operaotr처럼 사용할 수 있다.

r add s           r.add(s)
r less s          r.less(s)
r max s           r.max(s)

##### Relaxed Identifiers
operaotr는 identifier로 사용될 수 있다.
- 영문자: 문자로 시작하고, 뒤에는 문자 혹은 숫자가 올 수 있다.
- Symbolic: operator symbol로 시작해서, 다른 심볼이 뒤에 올 수 있다.
- `_` 문자는 문자로 카운트 된다
- 영문자 identifiers는 underscore로 끝날 수 있고 뒤에 다른 operator symbols가 붙을 수 있다.
* 만약 끝이 symbol들로 끝나면 뒤에 타입을 위한 `:` 과 한 칸 띄워야 한다.

examples
- x1
- *
- +?%&
- vector_++
- counter_=


-a 처럼 빼기가 아니라 마이너스 operator를 추가하고 싶다면 아래와 같이 해야 한다.
(`unary_` 가 앞에 붙어야 하고 `:`과 한칸 띄워 써야 한다.)
```scala
def unary_- : Rational = new Rational(-numer, denom)
```

#### Precedence Rules
연산자 우선순위.
첫 번째 문자에 따라 결정된다.
Java혹은 C와 차이 없다.

1번이 가장 낮은 순위이다.

1. (all letters)
2. |
3. ^
4. &
5. < >
6. = !
7. :
8. + -
9. * / %
10. (all other special values)


```scala
class Rational(x: Int, y: Int) {
  require(y != 0, "denominator must be nonezero")

  def this(x: Int) = this(x, 1) // 여기에서의 this는 constructor 의미로 쓰인다.

  private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
  private val g = gcd(x, y) // val로 선언했기에 바로 평가되서 다음 부턴 계산을 안하고 재사용한다.
  def numer = x / g
//    def numer = x / gcd(x,y) // 만약 이와 같이 선언 하면 매번 gcd를 계산해야 한다. 계산 리소스가 크고 가끔 호출될 때 사용하면 좋다.
  def denom = y / g

//    def less(that: Rational) = numer * that.denom < that.numer * denom
  def < (that: Rational) = numer * that.denom < that.numer * denom
  def max(that: Rational) = if (this.<(that)) that else this

  def +(r: Rational) =
    new Rational(numer * r.denom + r.numer * denom,
      denom * r.denom)

  def mul(r: Rational) =
    new Rational(numer * r.numer,
      denom * r.denom)

  def unary_- : Rational = new Rational(-numer, denom)

  def -(that: Rational) = this + -that

  override def toString = s"$numer/$denom"
}
```


## Week3

### Class Hierarchies
실제 메소드는 runtime type에 의존한다. 이를 dynamic binding이라고 한다.
이는 OOP에 기본 요소이다.

아래 함수는 abstract class이다.
```scala
abstract class IntSet {
  def incl(x: Int): IntSet
  def contains(x: Int): Boolean
}
```
추상 클래스는
- 구현체가 없는 멤버를 포함할 수 있다.
- new operator를 사용한 인스턴스 생성을 할 수 없다.

```scala
abstract class IntSet {
  def incl(x: Int): IntSet
  def contains(x: Int): Boolean
}

class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet {
  override def incl(x: Int): IntSet =
    if (x < elem) new NonEmpty(elem, left incl x, right)
    else if (x > elem) new NonEmpty(elem, left, right incl x)
    else this

  override def contains(x: Int): Boolean =
    if (x < elem) left contains x
    else if (x > elem) right contains x
    else true

  override def toString = "{" + left + elem + right + "}"
}

class Empty extends IntSet {
  override def incl(x: Int) = new NonEmpty(x, new Empty, new Empty)

  override def contains(x: Int) = false
  override def toString = "."
}


val t1 = new NonEmpty(3, new Empty, new Empty)
val t2 = t1 incl 4
```

위의 예에서 IntSet은 Empty와 NonEmpty의 superclass이다.
Empty와 NonEmpty는 IntSet의 subclasses이다.
스칼라에서 superclass가 없으면 java.lang에 있는 Java standard class Object 를 상속 받는다.
클래스의 direct or indirect superclass를 base classes라고 한다.
NonEmpty와 IntSet의 base classes는 Object이다.

non-abstract definition을 redfine할 때는 override keyword를 써줘야 한다.
```scala
abstract class Base {
  def foo = 1
  def bar: Int
}

class Sub extends Base {
  override def foo = 2
  def bar = 3
}
```

### Object Definitions
위의 예에서 유저가 많은 EmptySet을 만들게 되면 문제가 발생한다.
그래서 이를 object로 선언하는 것이 낫다.

```scala
object Empty extends IntSet {
  def incl(x: Int) = new NonEmpty(x, new Empty, new Empty)
  def contains(x: Int) = false
}
```
이렇게 하면 Empty라는 이름의 singleton object가 만들어 진다.
이로써 다른 Empty 인스턴스는 만들어질 수 없다.
Singleton Object는 values 이므로, Empty는 바로 평가된다.


### Programs
Scala에서 standalone application을 만드는 것은 가능하다.
main method를 포함하는 object를 만들면 된다.

```scala
object Hello {
  def main(args: Array[String]) = println("hello world!")
}
```
프로그램이 컴파일 되고 나면 아래의 커맨드로 실행할 수 있다.
```bash
> scala Hello
```

### Dynamic Binding
code invoked by a method call depends on the runtime of the object that contains the method

ex)
```scala
Empty contains 1
```
-> false

### Lecture 3.2 - How Classes Are Organized
#### Packages
Classes와 objects는 package안에 구성된다된

package에 속하는 class, object는 소스 파일의 최상단에 package를 써야 한다.
```scala
package progfun.examples

object Hello { ... }
```
아래와 같이 프로그램을 실행할 수 있다.
```bash
> scala progfun.examples.Hello
```

#### Forms of Imports
```scala
import week3.Rational // named imports
import week3.{Rational, Hello} // named imports
import week3._ // wildcard import
```

#### Automatic Imports
- All members of package scala
- All members of package java.lang
- All members of the singleton object scala.Predef


Int: scala.Int
Boolean: scala.Boolean
Object : java.lang.Object
require: scala.Predef.require
assert: scala.Predef.assert

#### Traits
Java 처럼 Scala는 오직 하나의 superclass를 가질 수 있다.(Single Inheritance)
하지만 여러개의 supertypes를 갖고 싶다면 어떻게 할까?
traits를 사용하면 된다.
trait은 abstract class처럼 정의 하면서 trait 키워드를 쓰면 된다를
```scala
trait Planar {
  def height: Int
  def width: Int
  def surface = height * width
}
```

```scala
class Square extends Shape with Planar with Movable ...
```
trait은 Java의 interface와 비슷하지만 더 강력하다.
fields와 concrete methods(정의된 메소드)를 가질 수 있기 때문이다.

하지만 trait은 (value) parameters를 가질 수 없다. 이는 클래스만 가능하다.

![](./scala class hierarchy.png)

#### Top Types
Any: The base type of all types. Methods: '==', '!=', 'equals', 'hashCode', 'toString
AnyRef: The base type of all reference types; Alias of 'java.lang.Object'
AnyVal: The base type of all primitive types

#### The Nothing Type
Nothing은 Scala type hierarchy에서 최 하단에 있다.
type Nothing에는 value가 없다.
왜 쓰일까?
- To signal abnormal termination
- As an element type of empty collection

#### Exceptions
자바와 유사하다
```scala
throw Exc
```
이 expr의 type은 Nothing이다.

example
```scala
def error(msg: String) = throw new Error(msg)

error("test")
```

#### The Null Type
every reference class type은 null 값을 갖는다.
null의 타입은 Null 이다.
Null은 Object를 상속받는 모든 클래스의 subtype이다.
하지만 AnyVal의 subtypes과는 incompativle 하다.

```scala
val x = null // x: Null
val y: String = null // y: String
val z: Int = null // error: type mismatch
```

### Lecture 3.3 - Polymorphism

#### Type Parameter
여러 타입에 대응할 수 있는 타입이다.

```scala
trait List[T]
class Cons[T](val head: T, val tail: List[T]) extends List[T]
class Nil[T] extends List[T]
```
타입 파라미터는 square brackets 안에 쓰인다.

#### Generic Functions
classes처럼 function에도 type parameter를 사용할 수 있다.
```scala
def singleton[T](elem: T) = new Cons[T](elem, new Nil[T])

singleton[Int](1)
singleton[Boolean](true)
```

#### Type Inference
스칼라는 function call의 arguments로 부터 parameter의 옳은 타입을 추측할 수 있다.
그렇기에 대부분의 경우에서 type parameters는 안써도 된다.

```scala
singleton(1)
singleton(true)
```

#### Type Evaluation
스칼라에서 Type parameter는 evaluation에 영향을 끼치지 않는다.
모든 type parameters와 type arguments는 프로그램을 평가하기 전에 제거된다.
이를 `type erasure`라고 부른다.

Java, Scala, Haskell, ML, OCaml에서는 type erasure를 사용한다.
하지만 C++, C#, F# 같은 언어는 run time시에도 type parameter를 유지한다.

#### Polymorphism
function type comes "in many forms"
- function이 여러 타입의 argument에 적용될 수 있다
- 타입이 여러 타입의 인스턴스를 가질 수 있다.

폴리몰피즘의 두 가지 형태
- subtyping: instances of subclass 는 base class로 전달 될 수 있다.
- generic: function 혹은 클래스의 instance는 type paramterization으로 만들 수 있다.