# longest-substring-without-repeating-characters

## Question
<a href="https://leetcode.cn/problems/longest-substring-without-repeating-characters"></a>

## Insight
- 滑动窗口
- 利用subString内无重复字符的性质实现滑动窗口
- 使用unordered_set或map记录是否出现重复字符

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> record;
        int ans = 0;
        int n = s.size();
        int rpointer = -1;
        

        for(int i = 0; i < n; ++i) {
            if(i > 0) {
                record.erase(s[i-1]);
            }

            while(rpointer + 1 < n && (!record.count(s[rpointer + 1]))) {
                ++rpointer;
                record.insert(s[rpointer]);
            }

            ans = max(ans, rpointer - i + 1);

        }

        return ans;

    }
};
```

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> record; //last time occurred
        int ans = 0;
        int n = s.size();
        int leftP = 0;
        int i;

        for(i = 0; i < n; ++i) {
            if((record.count(s[i]))) {
                ans = max(ans, i - leftP);
                leftP = max(leftP, record[s[i]] + 1);
            }
            record[s[i]] = i;

        }

        return max(ans, i - leftP);

    }
};
```