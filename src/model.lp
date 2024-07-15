\* Source Pyomo model name=unknown *\

min 
x1:
+ [
+2 x2 ^ 2
] / 2

s.t.

c_e_ONE_VAR_CONSTANT:
+1 x3
= 1

bounds
   1 <= x3 <= 1
   -inf <= x2 <= +inf
end
