
"""
Reserved words in C
"""

C89_RESERVED_WORDS = set(\
"""
auto
else
long
switch
break
enum
register
typedef
case
extern
return
union
char
float
short
unsigned
const
for
signed
void
continue
goto
sizeof
volatile
default
if
static
while
do
int
struct
_Packed
double
_Imaginary
""".split('\n'))

C99_RESERVED_WORDS = set(\
"""
inline
restrict
_Bool
_Complex
_Imaginary
""".split('\n'))

C11_RESERVED_WORDS = set(\
"""
_Alignas
_Alignof
_Atomic
_Generic
_Noreturn
_Static_assert
_Thread_local
""".split('\n'))

C23_RESERVED_WORDS = set(\
"""
alignas
alignof
bool
constexpr
false
nullptr
static_assert
thread_local
true
typeof
typeof_unqual
_BitInt
_Decimal128
_Decimal32
_Decimal64
""".split('\n'))

RESERVED_WORDS = C89_RESERVED_WORDS | C99_RESERVED_WORDS | C11_RESERVED_WORDS | C23_RESERVED_WORDS

