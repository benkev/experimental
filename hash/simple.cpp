#include <iostream>
#include <typeinfo>
using std::cout;
using std::endl;

int 
main ()
{
  cout << "Hello, world!\n";
  int a = 12;
  cout << typeof(a) << endl;
  cout << typeid(a).name() << endl;

  return 0;
}
