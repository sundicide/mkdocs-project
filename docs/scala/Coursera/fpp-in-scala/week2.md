# Higher order functions
pass functions as arguments and retun them as results.

functional languages treat functions as first-class values.
= like any other value, a function can be passed as a parameter and returned as a result.

provides a flexible way to compose program.



## Anonymous Function
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

# Currying

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


# Example: Finding Fixed Points
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

## functions as return values
위의 예제에서 평균을 통해 안정화시키는 기술은 추상화 될 수 있다.
```scala
def averageDamp(f: Double => Double)(x: Double) =
  (x + f(x)) / 2
def sqrt3(x: Double) =
  fixedPoint(averageDamp(y => x / y))(1)
```

Higher Order Function이 항상 옳은 것은 아니며 적절 할 때 사용해야 한다.


# Functions and Data
## Classes
```scala
class Rational(x: Int, y: Int):
  def numer = x
  def denom = y
```
위 정의는 two entities를 생성한다.
- Rational 이라는 이름의 new type
- 이 type의 element를 만들기 위한 Rational constructo

스칼라는 types과 value의 names를 `different namespace`에 보관하기 때문에 충돌을 걱정할 필요 없다.

## Objects
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

# More Fun With Rationals
Client's view에서는 내부가 어떻게 동작하던지 동일하게 보인다.

without affecting client를 하면서 다른 구현을 선택하는 것을 `data abstraction`이라고 한다.
S/E에서의 cornerstone이다.

## Self Reference
inside of a class, `this`는 현재 실행 중인 method내에서의 object를 의미한다

## Preconditions
`require`로 class에 조건을 추가할 수 있다.
조건에 맞지 않으면 IllegalArgumentException이 발생하며 추가한 에러 메세지가 출력된다.

## Assertions
require와 비슷한 의미이다.
require와 동일하게 condtion과 optional message string을 받는다.

```scala
val x = sqrt(y)
assert(x >= 0)
```

fail일 경우 assert는 require와 달리 AssertionError를 발생한다.

- require는 함수 호출자에게 precondition을 강요할 때 쓰인다
- assert는 함수 자신이 체크 할 때 사용한다.

## Constructors
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




# Evaluation and Operators
## Operators
### Infix Notation
parameter를 갖는 모든 메소드는 infix operaotr처럼 사용할 수 있다.

r add s           r.add(s)
r less s          r.less(s)
r max s           r.max(s)

### Relaxed Identifiers
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

## Precedence Rules
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
