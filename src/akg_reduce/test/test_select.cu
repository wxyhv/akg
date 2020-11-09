#include "./utils/util.cuh"
#include <typeinfo>
using namespace akg_reduce;
using namespace std;

int main() {
  typedef typename Select<false, double, float>::Type atype;
  atype a = 1.0;
  cout << typeid(a).name() << endl;

  typedef typename Select<(1 > 0), double, float>::Type btype;
  btype b = 1.0;
  cout << typeid(b).name() << endl;

  return 0;
}
