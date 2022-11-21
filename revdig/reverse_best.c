class Solution {
public:
    int reverse(int x) {
    
      int limit1=INT_MAX;
        int limit2=INT_MIN;
        int ans=0;
        while(x!=0){
          if((limit1/10<ans)||(limit1/10==ans&&x%10>limit1%10))return 0;
          if((limit2/10>ans)||(limit2/10==ans&&x%10<limit2%10))return 0;
        ans=ans*10+x%10;
        x/=10;
        }
        return ans;
        
    }
};
