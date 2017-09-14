#include "unity.h"
#include "dummy.h"

void test_foo (void)
{
  foo();
}

int main (void)
{
  UNITY_BEGIN ();
  RUN_TEST (test_foo);
  return UNITY_END ();
}
