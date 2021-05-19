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