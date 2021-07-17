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

## Read File

classpath 위치로 부터 파일 읽기
```scala
import scala.io.Source

def getResourcePath(): String = {
  val txtFile = Source.fromFile("src/main/resources/config.txt")
  txtFile.getLines().take(1).toList.head
}
```

resources폴더내 파일 읽기
```scala
import scala.io.Source

Option(getClass.getResourceAsStream("myfile.dat")) match {
  case None => sys.error("File is Empty")
  case Some(resource) => Source.fromInputStream(resource).getLines().toList
}

```
