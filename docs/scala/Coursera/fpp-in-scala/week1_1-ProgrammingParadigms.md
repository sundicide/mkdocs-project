# Lecture 1.1 - Programming Paradigms

functional Programming은 paradigm이다. classical imperative paradimg(Java or C)과 약간 다른.
scala에서는 이 2개의 paradigm을 합칠 수도 있다. 이는 다른 언어에서의 migration을 쉽게 해준다.

In science, a `paradigm` describes distinct concepts or thought patterns in some scientific discipline.

Main Programming Paradigms:
- imperative programming
- functional programming
- logic programming

object-oriented programming도 paradigm이라고 하는 사람들도 있지만 자신의 생각으로는 위 3개의 교차점에 있다고 생각한다.

## Imperative Programming
- modifying mutable variables
- using assignments
- and control structures such as if-then-else, loops, break, continue, return

Von Neumann computer의 sequence를 이해하는 것은 imperative program을 이해하는 most common informal way이다.

> Processor <------BUS ------> Memory


Problem: Scaling up. How can we avoid conceptualizing programs word by word?

high-level abstractions(collections, polynomials, geomtric shapes, strings, documents..)를 정의하는 테크닉이 필요하다.

Ideally: Develop theories of collections, shapes, strings, ...

## What is a theory
A theory consist of
- one or more data types
- operations on these types
- laws that describe the relationships between values and operations

보통 theory는 `mutations`를 describe하지 않는다!

mutation: identity는 유지하면서 something을 change하는 것이다.

### Theories without mutations
theory of polynomials

> (a*x + b) + (c*x + d) = (a+c)*x + (b+d)

theory of strings

> (a ++ b) ++ c = a ++ (b ++ c)


## Consequences for Programming
mathematical theroies를 따르면서 high-level concepts 구현을 하려면 mutation은 없어야 한다.
- theroies do not admit it
- mutation은 theories의 useful laws를 destoy 할 수 있다.

그러므로
- concentrate on defining theories for operators expressed as functions
- avoid mutations
- have powerful ways to abstract and compose functions

start of function programming means avoid mutations

## Functional Programming
- In a restricted sense, FP means programming without mutable variables, assignments, loops, and other imperative control structures
- In a wider sense, FP meas focusing on the functions
- In particular, functions can be valuses that are produced, consumed, and composed
- All this becomes easier in a functional language

## Functional Programming Language
- In a restricted sense, a functional programming language is one which does not have mutable variables, assignments, or imperative control structures.
- In a wider sense, a functional programming language enables the construction of elegant programs that focus on functions.
- In particular, functions in a FP language are first-class citizens. This means
    - they can be defined anywhere, including inside other functions
    - like any other value, they can be passed as parameters to functions and returned as results
    - as for other values, there exists a set operators to compose functions

## Some functional programming languages
In the restricted sense:
- Pure Lisp, XSLT, XPath, XQuery, FP
- Haskell (without I/O Monad or UnsafePerformIO)

In the wider sense:
- Lisp, Scheme, Racket, Clojure ▶ SML, Ocaml, F#
- Haskell (full language)
- Scala
- Smalltalk, Ruby (!)

## Why Functional Programming?
Functional Programming is becoming increasingly popular because it offers the following benfits.
- simpler reasoning principles
- better modularity
- good for exploiting parallelism for multicore and cloud computing.