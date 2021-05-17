# Week1-2: Recap: Collections

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