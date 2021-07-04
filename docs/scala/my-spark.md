
## _* 에 대하여

List(1,2,3,4)를 function(a, b, c, d)의 parameter로 전달하고 싶을 떈 어떻게 할까

`_*`를 이용하면 위와 같은 요구사항을 해결할 수 있다.

```scala
Row(line.head.toString :: line.tail.map(_.toDouble): _*)
```

## broadcast variables
Broadcast variables는 cluster내 각 node에 `오직 한 번`만 보내도록 한다. 또한 변수들을 cluster node내 메모리에 캐싱하여 프로그램 수행 시 사용할 수 있도록 한다.

```scala
val bcEmployees = sc.broadcast(employees)
```
