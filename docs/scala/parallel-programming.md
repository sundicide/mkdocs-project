# parallel programming

## Week1
Parallel computing은 many caculation을 동시에 computation 하는 것이다.

Basic principle: computation은 smaller subproblems로 나눌 수 있으며 각각은 동시에 solved 될 수 있다.
Assumption: parallel hardware가 있다면 이러한 computation을 parallel하게 실행할 수 있다.

20세기에 IBM이 first commercial parallel computers를 만들어 냈다.
그 당시에는 인기가 없었지만 현재는 computing performance가 증가하면서 인기가 많아졌다.

왜 Parallel Computing인가?
Parallel programming은 sequential programming보다 더 어렵다.
- 몇 몇 computation은 divide가 불가능하거나 어렵다.
- error 잡기 힘들다. 또 많은 새로운 타입의 에러가 발생한다.

`Speedup` is only reason why we bother paying for this complexity

Parallelism and concurrency 는 관련된 컨셉이다.

- Parallel Programming은 parallel hardware를 사용하여 계산을 더 빨리 수행한다. `Efficiency`가 최대 관심사이다. mainly concerned algorithmic problems, numerical computation or big data applications.  아래의 문제에 관심을 갖는다.
    - 어떻게 divide into sub problems 하고 어떻게 동시에 수행하느냐가 문제이다.
    - 최대한의 속도를 위해 어떻게 자원을 활용할까도 문제이다.
- Concurrent Program은 동시에 multiple execution을 수행할 때도 있지만 아닐 때도 있다. Modularity, responsiveness or maintainability가 최대 관심사다. targeted writing asynchronous applications. such as webservers, user interfaces or databases. 또한 아래의 문제에 관심을 갖는다.
    - 언제 계산을 수행해야 할까?
    - when and how 두 개의 concurrent execution이 정보를 교환해야 할까?
    - 어떻게 shared resources(file, db등) 접근을 관리해야 하는가?

두 개념이 공유하는 사항은 있지만 superset은 없다.


Parallelism manifests itself at diﬀerent granularity levels.
- bit-level parallelism: processing multiple bits of data in parallel
- instruction-level parallelism: executing diﬀerent instructions from the same instruction stream in parallel
- task-level parallelism: executing separate instruction streams in parallel

bit-level과 intstruction-level은 processor가 알아서 수행해준다.
우리가 이번에 다룰 것은 task-level이다.

parallel h/w에는 다양한 형태가 있다.
- multi-core processors
- symmetric multiprocessors(SMP): multiple identical processors가 bus와 연결되어 memory를 share하는 것이다.
- graphic processing unit: originally intended for graphics processing. 기본적으론 execute program을 하진 않지만 host processor에 의해서 수행할 수 있다.
- field-programmable gate arrays(FPGA): these are programmed in hardware description languages, such as Verilog and HTL
- computer clusters: groups of computers connected via a network. not sharing memory
이번 과정에서는 multi-core processors와 symmetric multiprocessor를 중점적으로 다룬다.



### Parallelism on the JVM I
There are many forms of parallelism in the wild.
From GPUs and custom parallel hardware, over multiprocessors, and multicore systems to distributed computer clusters.
Each parallel programming model is tailored to a specific use case, and has certain associated characteristics.

OS란? We will say that an operating system is a piece of software that manages hardware and software resources, and schedules program execution.
Process - OS에서 실행되는 program의 instance
same program은 하나의 process에서 여러 번 시작될 수 있고, same OS에서 동시에 수행될 수도 있다.
OS는 여러 process를 한정된 CPU에서 실행해야 한다. 그렇기 때문에 time slices of excution이 있다. 이러한 매커니즘을 multitasking이라고 한다.
그리고 각 process들은 메모리를 직접적으로 공유할 수 없다. = isolated memory area가 있다. (되는 OS가 있긴 하디)

그래서! Threads가 필요하다.
each process는 multiple independent concurrency units called threads를 갖는다.
using thread의 두 가지 장점이 있다.
- thread는 프로그램 내에서 programmatically started할 수 있다. = structure parallel computation을 만들기 쉽다. 프로세서보다.
- 더 중요한 것은 threads는 same memory address space를 공유한다.
각 thread는 program counter(current executed method의 메모리내 위치)와 program stack(memory내 공간으로서 수행할 methods를 갖는다.)을 갖는다.
JVM에서 실행되는 threads는 each other's program stacks를 modify 할 수 없다.

Each JVM process starts with a main thread.
보통의 프로그램은 main thread만 사용한다.
parallel programming에서는 계산을 위해 여러 threads를 사용한다.
additional threads를 사용하기 위해서는 아래의 스텝을 따라야 한다.
1. Define a Thread subclass
2. Instantiate a new Thread object.
3. Call start on the Thread object.

```scala
class HelloThread extends Thread {
  override def run() {
    println("Hello World!")
  }
}

val t = new HelloThread

t.start() // main thread와 HelloThread를 실행시킨다.
t.join() // join은 hellothread가 끝날때 까지 실행을 멈춘다.
```
위의 코드는 잘 동작하는 것 처럼 보인다.


```scala
class HelloThread2 extends Thread {
  override def run() {
    println("Hello")
    println("World!")
  }
}

def main(): Unit = {
  val t = new HelloThread2()
  val s = new HelloThread2()
  t.start()
  s.start()
  t.join()
  s.join()
}

main()
```
위의 코드는 수행할 때마다 다른 결과가 나올 수도 있다.
왜냐면 thread가 동시에 수행되기 때문에 어쩔때는 Hello Hello World World가 나올 것이고 어떤때는 Hello World Hello World가 나올 것이다.
(two threads can overlap)
하지만 우리는 때때로 Hello World 의 순서는 보장하고 싶을 수 있다(사이에 다른 thread가 실행되지 않도록).
Atomicity!
An operation is atomic if it appears as if it occurred instantaneously from the point of view of other threads.
예)
```scala
private var uidCount = 0L
def getUniqueId(): Long = {
  uidCount = uidCount + 1
  uidCount
}
```
위 함수는 uniqueId를 리턴하는 함수이다.
그렇기에 매번 고유한 값을 만들어 내야 하며 이를 위해 private var를 사용한다.
현재 이 함수는 atomic 하지 않다. 왜냐면 여러 쓰레드가 동시에 사용된다고 할 때 위에서 얘기했던 overlap이 발생할 수 있기 때문이다.
그렇게 되면 id가 1,2,3,4 순차적으로 가는것이 아니라 1,2,2,4,5,8,9 같이 될 수 있다.

```scala
var uidCount = 0L
def getUniqueId(): Long = {
  uidCount = uidCount + 1
  uidCount
}
def startThread() = {
  val t = new Thread {
    override def run(): Unit = {
      val uids = for (i <-0 until 10) yield getUniqueId()
      println(uids)
    }
  }
  t.start()
  t
}
startThread()
startThread()
```
이를 해결하기 위해 java와 scala는 synchronized block을 지원한다.
object x의 synchronized call 이후의 code block은 동시에 여러 쓰레드에서 실행될 수 없다.
(JVM에서는 synchronized object를 monitor라고 지정한다. 그리고 나서 다른 쓰레드가 접근하려고하면 block한다.)

```scala
private val x = new AnyRef {}
private var uidCount = 0L
def getUniqueId(): Long = x.synchronized {
  uidCount = uidCount + 1
  uidCount
}
```

### Parallelism on the JVM II
만약 은행 업무 같이 source와 target을 같이 block 해야 할 때는 기존의 synchronized를 사용할 수 없다.
이럴땐 more fine grained synchronization을 사용해야 한다.

```scala
class Account(private var amount: Int = 0) {
  def transfer(target: Account, n: Int) =
    this.synchronized {
      target.synchronized {
        this.amount -= n
        target.amount += n
      }
    }
}
```
A1이 source A2가 target이라고 할 때 thread는 A1에 monitor를 주고 A1,A2를 묶어서 monitor를 하나 또 준다.

```scala
class Account(private var amount: Int = 0) {
  def transfer(target: Account, n: Int) =
    this.synchronized {
      target.synchronized {
        this.amount -= n
        target.amount += n
      }
    }
}

def startThread(a: Account, b: Account, n: Int) = {
  val t = new Thread {
    override def run(): Unit = {
      for (i <- 0 until n) {
        a.transfer(b, 1)
      }
    }
  }
  t.start()
  t
}

val a1 = new Account(500000)
val a2 = new Account(700000)

val t = startThread(a1, a2, 150000)
val s = startThread(a2, a1, 150000)

t.join() // deadlock!!
s.join()
```
thread t가 끝나지 않으면서 deadlock이 걸린다.

deadlock이란 2개 혹은 그 이상의 threads들이 이미 점유중인 resources 놓지 않은 채 서로가 끝나길 기다리는 상황이다.

이를 해결하기 위한 한 가지 방법은 same order로 resource를 요청하는 것이다.
이때 resource에는 우선순위가 있다는 가정이다.

```scala
val uid = getUniqueUid()
private def lockAndTransfer(target: Account, n: Int) =
  this.synchronized {
    target.synchronized {
      this.amount -= n
      target.amount += n
    }
  }
  def transfer(target: Account, n: Int) =
    if (this.uid < target.uid) this.lockAndTransfer(target, n)
    else target.lockAndTransfer(this, -n)
```
위의 코드는 매 account object마다 uid를 부여해서 이에 따라 우선순위를 매기는 방법이다.

Memory Model이라는 것이 있다.
Memory Model이란 threads가 accessing shared memory를 interact 할 때의 규칙들을 설명해 놓은 것이다.
Java Memory Model은 memory model for JVM이다.
모든 룰을 다 설명할 순 없지만 기억해야 할 두 가지가 있다.
1. 메모리내 separate locations에 writing하는 two threads는 synchronization이 필요 없다.
2. 또 다른 thread Y를 호출하는 thread X는 join이 리턴된 후 thread Y의 모든 writes를 observe하도록 보장해야 한다.


### Running Computations in Parallel
Example: computing p-norm
vector (a1, a2)에 대한 2-norm은 다음과 같이 계산 한다.
> (a<sub>1</sub><sup>2</sup> + a<sub>2</sub><sup>2</sup>)<sup>1/2</sup>

array에 대한 sum of power는 아래와 같이 구할 수 있다.
```scala
def power(x: Int, p: Double): Int =
  math.exp(p * math.log(abs(x))).toInt

def sumSegment(a: Array[Int], p: Double, s: Int, t: Int): Int = {
  var i= s
  var sum: Int = 0
  while (i < t) {
    sum= sum + power(a(i), p)
    i= i + 1
  }
  sum
}
```
이 계산을 2 개로 나눌 수 있다.
0~n 까지의 합 + n~m 까지의 합

```scala
def pNormTwoPart(a: Array[Int], p: Double): Int = {
  val m = a.length / 2
  val (sum1, sum2) = (sumSegment(a, p, 0, m), sumSegment(a, p, m, a.length))
  power(sum1 + sum2, 1/p)
}
```
하지만 이는 sequential한 방법이다. sum1이 계산 다 되고 나서 sum2가 계산된다.
parallel로는 어떻게 할까?

```scala
def pNormTwoPart(a: Array[Int], p: Double): Int = {
  val m = a.length / 2
  val (sum1, sum2) = parallel(sumSegment(a, p, 0, m), sumSegment(a, p, m, a.length))
  power(sum1 + sum2, 1/p)
}
```
앞에 parallel을 붙이면 된다.
이렇게 되면 sum1과 sum2가 동시에 계산된다.

만약 4(2 + 2)개로 분리하면 어떨까?
```scala
val m1 = a.length/4
val m2 = a.length/2
val m3 = 3*a.length/4

val ((sum1, sum2),(sum3,sum4)) =
  parallel(
    parallel(sumSegment(a, p, 0, m1), sumSegment(a, p, m1, m2)),
    parallel(sumSegment(a, p, m2, m3), sumSegment(a, p, m3, a.length))
  )
```

generalize는 어떻게 할까?
recursion을 사용하면 가능하다!

```scala
def pNormRec(a: Array[Int], p: Double): Int =
  power(segmentRec(a, p, 0, a.length), 1/p)

// like sumSegment but parallel
def segmentRec(a: Array[Int], p: Double, s: Int, t: Int) = {
  if (t - s < threshold)
    sumSegment(a, p, s, t) // small segment: do it sequentially
  else {
    val m = s + (t - s)/2
    val (sum1, sum2) = parallel(segmentRec(a, p, s, m), segmentRec(a, p, m, t))
    sum1 + sum2
  }
}
```

그렇다면 signature of parallel은 어떻게 될까
```scala
def parallel[A, B](taskA: => A, taskB: => B): (A, B) = { ... }
```
- 항상 같은 value를 리턴해야 한다.
- (a,b) 보다 parallel(a, b)가 빠르다.
- by name으로 argumentds를 받는다.

왜 call by name으로 해야 할까?
```scala
def parallel1[A, B](taskA: A, taskB: B): (A, B) = { ...  }
val (va, vb) = parallel1(a, b)
```
위는 call by value버전이다.
이렇게 되면 즉시 평가가 되기 때문에 `val (va, vb) = (a, b)`가 바로 계산된다.
그렇기 때문에 이는 sequential하게 계산되버린 후 그 다음부터 parallel하게 계산된다.
(Because the parameters of parallel1 are call by value, a and b are evaluated sequentially in the second case, not in parallel as in the first case.)


다른 코드를 보자.
```scala
def sum1(a: Array[Int], p: Double, s: Int, t: Int): Int = {
  var i= s
  var sum: Int =
    0 while (i < t) {
      sum= sum + a(i) // no exponentiation!
      i= i + 1
    }
    sum
  }
  val ((sum1, sum2),(sum3,sum4)) =
    parallel(
      parallel(sum1(a, p, 0, m1), sum1(a, p, m1, m2)),
      parallel(sum1(a, p, m2, m3), sum1(a, p, m3, a.length))
    )
```
이 코드는 sumSegment와 다르게 speedup을 얻기가 힘들다. 왜 그럴까?
Memory Bottleneck!!

Array는 random access memory에 저장된다. 우리가 multiple processor를 갖고 있더라도 memory는 shared된다.
그 말인 즉슨 계산시간은 항상 메모리로부터 데이터를 fetch하는 속도 이상이 된다.

opportuniteis for spped-up을 고려할 때는 number of cores 뿐만 아니라 다른 shared resources(memory 등)에서 paralleism이 가능한지 알아야 한다.


### Monte Carlo Method to Estimate Pi

```scala
import scala.util.Random
def mcCount(iter: Int): Int = {
  val randomX = new Random
  val randomY = new Random
  var hits = 0
  for (i <- 0 until iter) {
    val x = randomX.nextDouble // in [0,1]
    val y = randomY.nextDouble // in [0,1]
    if (x*x + y*y < 1)
      hits = hits + 1
  }
  hits
}
def monteCarloPiSeq(iter: Int): Double =
  4.0 * mcCount(iter) / iter
```

parallel 버전
```scala
def monteCarloPiPar(iter: Int): Double = {
  val ((pi1, pi2), (pi3, pi4)) = parallel(
      parallel(mcCount(iter/4), mcCount(iter/4)),
      parallel(mcCount(iter/4), mcCount(iter - 3*(iter/4)))
    )
    4.0 * (pi1 + pi2 + pi3 + pi4) / iter
}
```

### First-Class Tasks
```scala
val (v1, v2) = parallel(e1, e2)
```
위 표현은 task를 이용해 다음과 같이 쓸 수 있다.

```scala
val t1 = task(e1)
val t2 = task(e2)
val v1 = t1.join
val v2 = t2.join
```
t = task(e)는 computation `e`를 `background`에서 start 함을 의미한다.
- t는 task이고, 계산 e를 수행한다.
- 현재 계산은 t로 parallel하게 진행된다.
- e의 결과를 얻고 싶을땐 `t.join`을 사용한다.
- t.join은 결과가 계산될 때까지 blocks and wait한다.

#### Task interface
```scala
def task(c: => A): Task[A]

trait Task[A] {
  def join: A
}
```
join을 implicit으로 변경하면 아래와 ㅏㄱㅌ다.
```scala
implicit def getJoin[T](x: Task[T]): T = x.join
```

```scala
val ((part1, part2),(part3,part4)) =
  parallel(
    parallel(sumSegment(a, p, 0, mid1), sumSegment(a, p, mid1, mid2)),
    parallel(sumSegment(a, p, mid2, mid3), sumSegment(a, p, mid3, a.length))
  )
  power(part1 + part2 + part3 + part4, 1/p)
```
위의 parallel p-form은 아래와 같이 task를 이용해서 변경할 수 있다.

```scala
val t1 = task {sumSegment(a, p, 0, mid1)}
val t2 = task {sumSegment(a, p, mid1, mid2)}
val t3 = task {sumSegment(a, p, mid2, mid3)}
val t4 = task {sumSegment(a, p, mid3, a.length)}
power(t1 + t2 + t3 + t4, 1/p)
```

그렇다면 parallel을 task를 이용해서 만들 수 있을까?

```scala
def parallel[A, B](cA: => A, cB: => B): (A, B) = {
  val tB: Task[B] = task { cB }
  val tA: A = cA
  (tA, tB.join)
}
```

아래는 잘못된 버전이다.
```scala
// WRONG
def parallelWrong[A, B](cA: => A, cB: => B): (A, B) = {
  val tB: B = (task { cB }).join
  val tA: A = cA
  (tA, tB.join)
}
```
무슨 일이 일어날까?
cA와 cB가 parallel하게 실행되지 않고 tB가 먼저 계산되기를 기다린 후 tA를 계산하게 된다.

### How Fast are Parallel Programs?
어떻게 performance를 측정할 수 있을까?
- empirical measurement
- asymptotic analysis

Asymptotic anlysis는 아래의 경우에 algorithms scale을 어떻게 할지 이해하는데 중요하다.
- inputs get larger
- we have more h/w parallelism available
worst-case bounds를 알 수 있다.

### Testing and Benchmarking
- testing: 프로그램의 일부분이 의도한 대로 동작하는 지를 알기 위함
맞는지 아닌지를 판단하는 binary(true / false) 결과물을 만들어 낸다.
- benchmarking: 프로그램 일부분의 performance metrics을 계산함
continuous value(소요 시간 등)를 리턴한다.

왜 Benchmarking을 해야 하느냐?
Parallel programs는 speed up이 가장 중요하다.
그렇기에 sequential programs보다 benchmarking이 중요하다.

Performance(특히나 running time)는 많은 요소에 관련있다.
- processor speed
- number of processor
- memory access latency and throughput (affects contention)
- cache behavior(e.g. false sharing, associativity effects)
- runtime behavior(e.g. garbage collection, JIT compilation, thread scheduling)

그래서 measuring performance는 정확히 알아내기 힘들다.
이를 methodoliges를 만들면 아래와 같이 한다.
- 여러번 반복해서 수행한다.
- 통계를 낸다: mean, variance
- 아웃라이어를 제거한다.
- ensuring steady state(warm-up)
- preventing anomalies(GC, JIT compilation, aggresive optimizations)

ScalaMeter는 JVM 을 위해 benchmarking과 performance regression testing을 해준다.
- performance regression testing: 이전의 run과 비교해서 performance를 비교
- benchmarking: 현재 프로그램 혹은 프로그램의 일부분의 performance를 측정
이번 강좌에서는 benchmarking에 집중한다.

#### Using ScalaMeter
first, add dependency
```scala
libraryDependencies += "com.storm-enroute" %% "scalameter-core" % "0.6"
```

그리고 이를 import해서 사용
```scala
import org.scalameter._

val time = measure {
  (0 until 1000000).toArray
}

println(s"Array initialization time: $time ms")
```

#### JVM Warmup
위의 예는 two consecutive runs of program 할 때 different running times를 보였다
JVM program이 시작할 때 warmup을 하는 시간이 필요하다. 그리고 그 이후에는 maimum performance를 발휘한다.
- 첫 번쨰로 프로그램이 interpreted mode로 run 된다.
- 그리고 프로그램의 일부분이 machine code로 compile된다.
- 후에 JVM은 additional dynamic optimizations를 적용할 지를 결정한다.
- eventually, 프로그램이 steady state로 된다.

우리는 항상 steady state program performance(warm up이 된 후에 프로그램 속도)를 측정하고 싶다.
ScalaMeter의 Warmer objects는 이를 benchmark 할 수 있다.
```scala
import org.scalameter._
val time = withWarmer(new Warmer.Default) measure {
  (0 until 1000000).toArray
}
```

configuration도 추가할 수 있다.
```scala
val time = config(
  Key.exec.minWarmupRuns -> 20,
  Key.exec.maxWarmupRuns -> 60,
  Key.verbose -> true
) withWarmer(new Warmer.Default) measure {
  (0 until 1000000).toArray
}
```

ScalaMeter는 running time이외에도 아래의 리스트에 속하는 것들을 측정할 수 있다.
- Measurer.Default – plain running time
- IgnoringGC – running time without GC pauses
- OutlierElimination – removes statistical outliers
- MemoryFootprint – memory footprint of an object. object가 memory에서 얼마만큼의 자리를 차지하는지 측정
- GarbageCollectionCycles – total number of GC pauses
- newer ScalaMeter versions can also measure method invocation counts and boxing counts

```scala
val time = withWarmer(new Measurer.MemoryFootprint) measure {
  (0 until 1000000).toArray
}
```
steady state이전일때는 값이 -로 나올 수 있다.
4000근처의 값이 나오는데 이는 4000KB = 4MB를 차지한다는 의미이다.



## Week2
fold와 reduce의 차이는 초기값을 갖느냐 아니면 빈 List를 초기값으로 사용하느냐의 차이다.
left와 Right의 차이는 operation을 어느 방향으로 적용할 것인지의 차이다.
```scala
List(1,3,8).foldLeft(100)((s,x) => s - x) == ((100 - 1) - 3) - 8 == 88
List(1,3,8).foldRight(100)((s,x) => s - x) == 1 - (3 - (8-100)) == -94
List(1,3,8).reduceLeft((s,x) => s - x) == (1 - 3) - 8 == -10
List(1,3,8).reduceRight((s,x) => s - x) == 1 - (3 - 8) == 6
```


sequential version of scanLeft
```scala
def scanLeft[A](inp: Array[A], a0: A, f: (A,A) => A, out: Array[A]): Unit = {
  out(0)= a0
  var a = a0
  var i= 0
  while (i < inp.length) {
    a= f(a,inp(i))
    i= i + 1
    out(i)= a
    }
}
```

```scala
def scanLeft[A](inp: Array[A], a0: A, f: (A,A) => A, out: Array[A]) = {
  val fi = { (i:Int,v:A) => reduceSeg1(inp, 0, i, a0, f) }
  mapSeg(inp, 0, inp.length, fi, out)
  val last = inp.length - 1
  out(last + 1) = f(out(last), inp(last))
}
```

## Week3 Data-Parallelism
Previously, we learned about task-parallel programming.
> A form of parallelization that distributes execution processes across computing nodes.

We know how to express parallel programs with task and parallel constructs.
Next, we learn about the data-parallel programming.
> A form of parallelization that distributes data across computing nodes.

The simplest form of data-parallel programming is the parallel for loop.

```scala
def initializeArray(xs: Array[Int])(v: Int): Unit = {
  for (i <- (0 until xs.length).par) {
    xs(i) = v
  }
}
```
Range에 `.par`를 붙이면 `parallel range`로 변환된다.
parallel loop는 different processors에서 concurrently with each other로 실행될 것이다
parallel loop는 어떠한 값도 리턴하지 않는다
그렇기에 이와 interact할 수 있는 유일한 방법은 assign밖에 없다. 이는 side effect를 유발할 수 있음을 의미한다. = not very functional 하다.


*Workload* 란 각각의 input element에 대해서 이를 실행하는데 필요한 amount of work 이다.
different data-parallel programs는 different workloads를 갖는다.
(= input element에 따라서 수행량이 다르다.)
*data-parallel scheduler* 는 이런 different workloads를 efficiently balance 해주기 때문에 프로그래머가 이를 직접 관리할 필요는 없다.

Scala에서 대부분의 collection operation은 data-parallel을 지원한다.
```scala
(1 until 1000).par
  .filter(n => n % 3 == 0)
  .count(n => n.toString == n.toString.reverse)
```
filter와 count도 data parallel operation을 지원한다.

하지만 몇몇 collection operations는 parallelizable 하지 않다.

foldLeft를 이용한 sum을 생각해보자.

```scala
def sum(xs: Array[Int]): Int = {
  xs.par.foldLeft(0)(_ + _)
}
```
이는 parallel 하게 수행되는가?
아니다.

먼저 foldLeft의 signature를 살펴보자.
```scala
def foldLeft[B](z: B)(f: (B, A) => B): B
```
함수 f를 보면 A와 B 타입을 받아서 B 타입으로 만들어 준다.
이 의미는 foldLeft를 연속적으로 수행할 때 먼저 수행한 foldLeft의 결과를 받고 나서야 다음 foldLeft를 계산할 수 있음을 의미한다.(= 참조 투명하지 않다.)

이와 같은 종류의 foldRight, reduceLeft, reduceRight, scanLeft and scanRight는 동일한 문제를 갖는다.

만약 이를 parallel하게 하고 싶다면 아래와 같은 정의가 필요하다.

```scala
def fold(z: A)(f: (A, A) => A): A
```

이를 이용하면 max를 아래와 같이 구현할 수 잇다.
```scala
def max(xs: Array[Int]): Int = {
  xs.par.fold(Int.MinValue)(math.max)
}
```

data-parallel operation을 수행할 때는 몇 가지 주의 사항이 있다.

1. same memory locations를 수정하는 것을 피해야 한다.

```scala
def intersection(a: GenSet[Int], b: GenSet[Int]): Set[Int] = {
  val result = mutable.Set[Int]()
  for (x <- a) if (b contains x) result += x
  result
}

intersection((0 until 1000).toSet, (0 until 1000 by 4).toSet)
intersection((0 until 1000).par.toSet, (0 until 1000 by 4).par.toSet)
```
위에서 .par로 실행한 코드는 parallel하게 동작하면서 result라는 Set을 변경한다.
비록 여기에선 오류가 발생하지 않았지만 이는 side effect를 초래할 수 있다.

이를 해결하기 위해선 아래와 같은 방법이 있다.

1\) use a concurrent collection, which can be mutated by multiple threads
```scala
import java.util.concurrent._
def intersection(a: GenSet[Int], b: GenSet[Int]) = {
val result = new ConcurrentSkipListSet[Int]()
for (x <- a) if (b contains x) result += x result
}
intersection((0 until 1000).toSet, (0 until 1000 by 4).toSet)
intersection((0 until 1000).par.toSet, (0 until 1000 by 4).par.toSet)
```

2\) correct combinator를 사용한다.
여기에서는 filter를 사용하면 된다.

```scala
def intersection(a: GenSet[Int], b: GenSet[Int]): GenSet[Int] = {
  if (a.size < b.size) a.filter(b(_))
  else b.filter(a(_))
}
intersection((0 until 1000).toSet, (0 until 1000 by 4).toSet)
intersection((0 until 1000).par.toSet, (0 until 1000 by 4).par.toSet)
```


2. data-parallel operation 수행 중일 때 parallel collection을 수정하면 절대 안된다. 또한 수정된 parallel collection을 읽는 것도 절대 안된다.

```scala
val graph = mutable.Map[Int, Int]() ++= (0 until 100000).map(i => (i, i + 1)) graph(graph.size - 1) = 0
for ((k, v) <- graph.par) graph(k) = graph(v)
val violation = graph.find({
  case (i, v) => v != (i + 2) % graph.size
})
println(s”violation: $violation”)
```
위 코드는 프로세스를 진행하면서 collection을 수정한다.


위의 오류를 피하기 위해 아래의 collection을 사용할 수 있다.
TrieMap Collection은 이런 rule에 대해 exception을 갖고 있다.

```scala
val graph = concurrent.TrieMap[Int, Int]() ++= (0 until 100000).map(i => (i, i + 1))
graph(graph.size - 1) = 0
val previous = graph.snapshot()
for ((k, v) <- graph.par)
  graph(k) = previous(v)

val violation = graph.find({
  case (i, v) => v != (i + 2) % graph.size
})
println(s”violation: $violation”)
```
