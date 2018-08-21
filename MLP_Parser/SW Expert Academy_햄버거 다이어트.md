```c++
# DFS
#include <iostream>
#include <algorithm>

using namespace std;

int n, limit;
int score[21], k[21];
int total;

void dfs(int cur, int depth, int sum_score, int sum_k) {

	if (depth == n || sum_k > limit) {
		return;
	}

	total = max(total, sum_score);

	for (int i = cur + 1; i < n; i++) {
		dfs(i, depth + 1, sum_score + score[i], sum_k + k[i]);
	}

}

int main(int argc, char** argv)
{
	int test_case;
	int T;

	//freopen("input.txt", "r", stdin);
	cin >> T;
	/*
	여러 개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
	*/
	for (test_case = 1; test_case <= T; ++test_case)
	{
		total = 0;
		scanf("%d %d", &n, &limit);

		for (int i = 0; i < n; i++) {
			scanf("%d %d", &score[i], &k[i]);
		}

		for (int i = i = 0; i < n; i++) {
			dfs(i, 0, score[i], k[i]);
		}


		printf("#%d %d\n",test_case, total);
	}
	return 0;//정상종료시 반드시 0을 리턴해야합니다.
}
```

