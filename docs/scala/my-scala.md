# My Scala
## Set
- `Set`은 `Seq`보다 random lookup이 빠르다.

- `++`를 사용하면 Set에 multiple variables를 추가할 수 있다.
```scala
val sets = Set() ++ (
  for {
    num <- numsList
  } yield num
)
```
