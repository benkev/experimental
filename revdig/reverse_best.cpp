// https://leetcode.com/problems/reverse-integer/
//
// $ g++ reverse_best.cpp -o reverse_best
//          or
// $ g++ -std=c++11 reverse_best.cpp -o reverse_best
//


#include <climits>
#include <iostream>
//#include <string>

using namespace std;

class Solution {
public:
    int reverse(int x) {
    
        int limit1=INT_MAX;
        int limit2=INT_MIN;
        int ans=0;
        
        while(x != 0){
            
            if ((limit1/10 < ans) || (limit1/10 == ans && x%10 > limit1%10))
                return 0;
            if ((limit2/10 > ans) || (limit2/10 == ans && x%10 < limit2%10))
                return 0;

            ans=ans*10+x%10;
            x/=10;
        }

        return ans;
    }
};


int main(int argc, char *argv[]) {

    Solution revdig;
    int x = stoi(argv[1]);
    
    int y = revdig.reverse(x);

    std::cout << x << "==>>" << y << ".\n";
}
