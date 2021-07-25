# 02 - Rust Program Structure

```rust
fn main() {
    println!("Hello, world!");
}
```

## main 함수
모든 Rust 프로그램에는 이름이 main인 함수가 한 개 있어야 합니다.
main 함수의 코드는 모든 Rust 프로그램에는 이름이 main인 함수가 한 개 있어야 합니다.

## 함수와 인수
Rust에서 함수를 선언하기 위해 fn 키워드를 사용합니다. 함수 이름 다음에 함수가 입력으로 예상하는 매개 변수 또는 인수 개수를 컴파일러에 알립니다

## 코드 들여쓰기
함수 본문에서 대부분의 코드 문은 세미콜론 ;으로 끝납니다. Rust는 이러한 문을 순서대로 처리합니다. 코드 문이 세미콜론으로 끝나지 않으면 Rust는 시작 문이 완료되기 전에 그 다음 코드 줄을 실행해야 한다고 인식합니다.

## println! 매크로
매크로는 함수와 같으며 입력 인수의 개수가 가변적입니다.
println! 매크로는 하나 이상의 입력 인수를 예상하고 이것을 화면 또는 표준 출력에 표시합니다.

```rust
fn main() {
    // Our main function does one task: call the println! macro
    // println! displays the input "Hello, world!" to the screen
    println!("Hello, world!");
}
```

## {} 인수의 값 대체
`println!` 매크로는 텍스트 문자열 안의 각 중괄호 `{}` 인스턴스를 목록의 그 다음 인수 값으로 바꿉니다.

```rust
fn main() {
    // Call println! with three arguments: a string, a value, a value
    println!("The first letter of the English alphabet is {} and the last letter is {}.", 'A', 'Z');
}
```

출력
```
The first letter of the English alphabet is A and the last letter is Z.
```

## variables
변수는 `let`으로 선언한다

```rust
let a_number;
let a_number = 10;
```

## 변경 불가 및 변경 가능
Rust에서는 기본적으로 변수 바인딩을 변경할 수 없습니다. 변수를 변경할 수 없는 경우 값을 이름에 바인딩한 후에는 해당 값을 변경할 수 없습니다.

예를 들어 이전 예제의 a_number 변수 값을 변경하려고 하면 컴파일러에서 오류 메시지를 표시합니다.

값을 변경하려면 먼저 mut 키워드를 사용해 변수 바인딩을 변경할 수 있게 해야 합니다.

```rust
// The `mut` keyword lets the variable be changed
let mut a_number = 10; 
println!("The number is {}.", a_number);

// Change the value of an immutable variable
a_number = 15;
println!("Now the number is {}.", a_number);
```

## 변수 섀도잉
기존 변수와 동일한 이름을 사용하는 새 변수를 선언할 수 있습니다. 새 선언으로 새 바인딩이 생성됩니다. 새 변수는 이전 변수를 섀도잉하므로 Rust에서는 이 작업을 "섀도잉"이라고 합니다. 이전 변수는 여전히 존재하지만 이 범위에서는 더 이상 참조할 수 없습니다.

다음 코드는 섀도잉 사용을 보여줍니다. 이름이 shadow_num인 변수를 선언합니다. 각 let 작업으로 이전 변수 바인딩을 섀도잉하는 동안 number라는 새 변수를 생성하므로 변수를 변경 가능으로 정의하지 않습니다.

```rust
// Declare first variable binding with name "shadow_num"
let shadow_num = 5;

// Declare second variable binding, shadows existing variable "shadow_num" 
let shadow_num = shadow_num + 5; 

// Declare third variable binding, shadows second binding of variable "shadow_num"
let shadow_num = shadow_num * 2; 

println!("The number is {}.", shadow_num);
// 20
```

## 데이터 형식
Rust는 정적으로 형식화된 언어입니다. 컴파일러는 프로그램이 컴파일하고 실행할 코드에 있는 모든 변수의 정확한 데이터 형식을 알아야 합니다.

컴파일러는 일반적으로 바인딩된 값에 따라 변수의 데이터 형식을 유추할 수 있습니다. 코드에 형식을 항상 명시할 필요는 없습니다. 여러 형식을 사용할 수 있는 경우 형식 주석 을 사용하여 컴파일러에 특정 형식을 알려야 합니다.

다음 예에서는 number 변수를 32비트 정수로 만들도록 컴파일러에 지시합니다. 변수 이름 뒤에 데이터 형식 u32를 지정합니다. 변수 이름 뒤에 콜론 :을 사용합니다.

```rust
let number: u32 = 14;
println!("The number is {}.", number);

let number: u32 = "14"; // text는 u32 type이 아니므로 compiler error 발생
```

### 기본 제공 데이터 형식
- 정수
- 부동 소수점 숫자
- 부울
- 문자

```rust
let number_64 = 4.0;      // compiler infers the value to use the default type f64
let number_32: f32 = 5.0; // type f32 specified via annotation
```

```rust
// Declare variable to store result of "greater than" test, Is 1 > 4? -- false
let is_bigger = 1 > 4;
println!("Is 1 > 4? {}", is_bigger);
```

char 형식은 가장 기본적인 텍스트 형식입니다. 그 값은 항목을 작은 따옴표로 묶어 지정합니다.
```rust
let uppercase_s = 'S';
let lowercase_f = 'f';
let smiley_face = '😃';
```

String과 &str 간의 차이점을 완전히 이해하려면 Rust의 소유권 및 대여 시스템에 대해 알아보아야 합니다. 그때까지는 String 형식 데이터를 프로그램이 실행될 때 변경될 수 있는 텍스트 데이터로 생각할 수 있습니다. &str 참조는 프로그램이 실행될 때 변하지 않는 텍스트 데이터에 대한 변경 불가능한 보기입니다.
(실제로 Rust에는 두 개 이상의 문자열 형식이 있습니다. 이 모듈에서는 String 및 &str 형식만 다룹니다. Rust 설명서에 나오는 문자열 형식에 대해 자세히 알아볼 수 있습니다.)

```rust
// Specify the data type "char"
let character_1: char = 'S';
let character_2: char = 'f';
   
// Complier interprets a single item in quotations as the "char" data type
let smiley_face = '😃';

// Complier interprets a series of items in quotations as a "str" data type and creates a "&str" reference
let string_1 = "miley ";

// Specify the data type "str" with the reference syntax "&str"
let string_2: &str = "ace";

println!("{} is a {}{}{}{}.", smiley_face, character_1, string_1, character_2, string_2);

// 😃 is a Smiley face.
```

### 튜플
튜플은 하나의 복합 값으로 수집되는 다양한 형식의 값을 그룹화한 것입니다. 튜플의 개별 값을 요소라고 합니다. 그 값은 괄호 (<value>, <value>, ...)로 묶은 쉼표로 구분된 목록으로 지정합니다.

튜플에는 요소의 수와 동일한 고정 길이가 있습니다. 튜플이 선언된 후에는 크기가 커지거나 축소될 수 없습니다. 요소는 추가하거나 제거할 수 없습니다. 튜플의 데이터 형식은 요소의 데이터 형식 시퀀스로 정의됩니다.

```rust
// Tuple of length 3
let tuple_e = ('e', 5i32, true);
```

이 튜플의 형식 서명은 세 가지 요소의 형식 시퀀스 (char, i32, bool)로 정의됩니다.

튜플의 요소는 0부터 시작하는 인덱스 위치에서 액세스할 수 있습니다. 이 프로세스를 튜플 인덱싱이라고 합니다. 튜플의 요소에 액세스할 때 구문 <tuple>.<index>을 사용합니다.

다음 예제는 인덱싱을 사용하여 튜플의 요소에 액세스하는 방법입니다.

```rust
// Declare a tuple of three elements
let tuple_e = ('E', 5i32, true);

// Use tuple indexing and show the values of the elements in the tuple
println!("Is '{}' the {}th letter of the alphabet? {}", tuple_e.0, tuple_e.1, tuple_e.2);

// Is 'E' the 5th letter of the alphabet? true
```

