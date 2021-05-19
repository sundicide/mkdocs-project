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

## Value Definitions
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

## Nested Functions
small func로 분리하는 것. good FP styles
sqrtIter, imporve 같은 함수들은 외부에 공개(direct 호출) 하고 싶지 않을 수 있다.

이러한 보조 함수들을 내부 함수로 둠으로써 name-space pollution을 방지할 수 있다.

```scala
def sqrt(x: Double) = {
  def improve
  def sqrtIter
}
```

## Lexical Scoping
outer block에 있는 definitions는 inside block에서 visible하다.

보통 문장 라인 끝 `;`는 optional이다.
다만 한 문장에 여러 expr을 표현할 때는 필수 이다.

```scala
val y = x + 1; y + y
```

## Tail Recursion
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

