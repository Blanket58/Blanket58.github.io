# Debug

## 查看调用栈

`traceback()`，调用栈(call stack)，显示导致错误的调用顺序，由下向上阅读调用栈，初始调用在最下方。它只告诉我们问题在哪里发生，而不会告诉我们为什么会出错。

## Debug模式



```R
options(error = browser)
```

在错误发生的地方打开一个交互式对话，`browser[1]>`的位置输入以下命令：

- `Next  n`：执行函数的下一步
- `Stop info  s`：与执行下一步类似，但如果下一步是一个函数，就单步执行这个函数
- `Finish  f`：结束当前循环或函数的执行
- `Continue  c`：离开交互式调试并继续函数的正常执行
- `Stop  Q`：停止调试，终止函数
- `Enter`：重复前一个命令，它很容易被我们不小心地激活，所以可以使用`options(browserNLdisabled = TRUE)` 这一项在帮助文档中没有找到，可能在新版本的R中已经被删除
- `where`：输出当前调用的栈追踪（等价于交互式的traceback）

```R
options(error = recover)
```

这是browser的增强版，可以让我们在交互环境下选择进入调用栈中的某一个。

## 服务器环境Debug

某些情况下，例：代码在服务器上运行出错时，此时没有一个IDE可供使用，可以先将错误信息存在一个rda文件里，随后在rstudio中load它，再使用`debugger()`调试代码

```R
# In batch R process ----
dump_and_quit <- function() {
  dump.frames(to.file = TRUE)
  q(status = 1)
}
options(error = dump_and_quit)
```

- 这里运行程序，在遇到错误时R会生成一个last.dump.rda文件并自动退出
- 用于非交互使用的默认错误处理程序会调用`q(“no”，1，false)`并返回错误状态1
- 这个状态是可以自定义的，例如定义一个条件发生时R退出`if () quit(status = 100)`，这里的状态码可以自定义
- 在定义了退出附带的状态码之后，当R程序以脚本从外部执行时，如果出错，会向操作系统返回这个状态码

```R
# In a later interactive session
load("last.dump.rda")
debugger()
```

记得调试完代码后将错误行为重置为默认状态。

```R
options(error = NULL)
```

## Warning

```R
options(warn = 2)
```


函数可能产生一个意想不到的警告(warnings)，查找警告最简单的方法就是通过设置上面代码将其转变成错误，再使用`traceback()`查看调用栈；

在调用栈中会看到一些函数：

- `doWithOneRestart()`

- `withOneRestart()`

- `withRestarts`

- `.signalSimpleWarning()`

这些函数只是用来将警告变成错误的，不用理会。

## Message

函数还有可能产生一个意想不到的信息(message)，R没有内置的方法让我们将消息转变成错误，不过可以自己编写一个：

```R
message2error <- function(code) {
  withCallingHandlers(code, message = function(e) stop(e))
}

# 测试代码如下：
f <- function() g()
g <- function() message("Hi!")
f()
#> Hi!
```

我们不知道这个message是怎么来的，将其转变成错误再查看调用栈来搞清楚

```R
message2error(f())
traceback()
```

只适用于将message转error，对于cat()或print()的输出，我暂时没有找到可以用的调试方法。

## 处理错误

有三个处理错误条件编程的工具：

### try()

可以忽略错误继续执行下面的代码，通过设置`try(..., silent = TRUE)`来禁止报错信息的显示，如果需要在`try()`中写入大段代码块，将它们包裹在`{}`里，例：

```R
try({
  x <- 1
  b <- "x"
  a + b
})
```

将`try()`的输出绑定到一个对象时，如果代码执行错误，这个对象就是一个`try-error`类，没有内置函数可以处理这个类，可以自定义一个用来定位错误发生位置的函数

```R
is.error <- function(x) inherits(x, "try-error") 
```

这里判断对象`x`是否继承了`try-error`类，默认参数`which = FALSE`的情况下，如果的确继承，则函数返回逻辑值`TRUE`

```R
succeeded <- !sapply(results, is.error)
```

`try()`还可以用来在表达式失败时使用默认值，实现方式如下：

```R
default <- NULL
try(default <- read.csv("possibly-bad-input.csv"), silent = TRUE)
```

同样可以用`plyr::failwith(default = NULL, f, quiet = TRUE)`来实现。

### tryCatch()

```R
try2 <- function(code, silent = FALSE) {
  tryCatch(code, error = function(c) {
    msg <- conditionMessage(c)
    if (!silent) message(c)
    invisible(structure(msg, class = "try-error"))
  })
}
```

上面的函数用`tryCatch()`实现了`try()`的功能，其中`conditionMessage()`是用来从`condition`中提取`message`的，在函数的最后定义了一个不可见的S3类下的对象。

`tryCatch()`还有一个参数`finally`，它与`on.exit()`功能类似。

### withCallingHandlers()

`tryCatch()`处理程序的返回值是由`tryCatch()`返回的，而`withCallingHandlers()`处理程序的返回值是被忽略的。

*The handlers in `withCallingHandlers()` are called in the context of the call.
The handlers in `tryCatch()` are called in the context of tryCatch().*

这就会导致查看调用栈时，它们的输出是不同的，但通常没有必要使用`withCallingHandlers()`，除非希望更精确地捕获什么出错了并将其传递给其他对象。

### 自定义信号类

大多数函数运行出错时都只会调用一个`stop()`，输出一段字符串，这对于我们想具体地识别错误有时具有挑战性。

如果我们想区分不同类型的错误，可以定义自己的类，每一个信号函数`stop()`、`warning()`、`message()`都接受一个字符串或一个condition S3对象作为它们的输入。

R没有内置函数来构造conditions，我们可以自己写一个，conditions必须包含message和call元素，且必须继承condition类和error、warning、message类中的任意一个：

```R
condition <- function(subclass, message, call = sys.call(-1), ...) {
  structure(
    list(message = message, call = call),
    class = c(subclass, "condition"),
    ...
  )
}
is.condition <- function(x) inherits(x, "condition")

# 自定义my_error类继承error类与condition类
e <- condition(c("my_error", "error"), "This is an error")
stop(e)
#> Error: This is an error

# 自定义my_warning类继承warning类与condition类
w <- condition(c("my_warning", "warning"), "This is a warning")
warning(w)
#> Warning message: This is a warning

# 自定义my_message类继承message类与condition类
m <- condition(c("my_message", "message"), "This is a message")
message(m)
#> This is a message
```

下来就可以使用`tryCatch()`对不同的错误采取不同的行动

```R
custom_stop <- function(subclass, message, call = sys.call(-1), ...) {
  c <- condition(c(subclass, "error"), message, call = call, ...)
  stop(c)
}
my_log <- function(x) {
  if(!is.numeric(x))
    custom_stop("invalid_class", "my_log() needs numeric input")
  if(any(x < 0))
    custom_stop("invalid_value", "my_log() needs positive input")
  log(x)
}

# tryCatch()中可以针对每个自定义的条件信号类设置handler
tryCatch(
  my_log("a"),
  invalid_class = function(c) "invalid class",
  invalid_value = function(c) "invalid value"
)
```

但需要注意的是，`tryCatch()`在匹配handler时，会去匹配与条件信号类继承关系中任意类相匹配的第一个handler，而不会去匹配类名完全一致的那一个handler，所以在handler的放置顺序上要注意。举个例子：

invalid_class类的继承关系是 invalid_class -> error -> condition，那么如果像下面这样就会出错

```
tryCatch(
  my_log(-1),
  condition = function(c) "invalid class",
  invalid_value = function(c) "invalid value"
)
#> [1] "invalid class"
```

这也是R的S3没有真正意义上的类继承关系所导致，所谓的类继承只是将所有类名依次放在`class()`属性的字符向量中。
