#include <iostream>
#include <vector>
using namespace std;
typedef struct tree {
	int head = -1;
	int isChecked = 0;
	int me;
	vector<int> member;
}tree;
void makeTree(int me, int head);
tree *t;
int n, member[2];
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	cin >> n;
	t = new tree[n + 1];
	cin.ignore();
	for (int i = 1; i < n; i++) {
		cin >> member[0] >> member[1];
		t[member[0]].me = member[0];
		t[member[0]].member.push_back(member[1]);
		t[member[1]].me = member[1];
		t[member[1]].member.push_back(member[0]);
	}
	makeTree(1, 0);
	for (int i = 2; i <= n; i++) {
		cout << t[i].head << "\n";
	}
	delete[] t;
	
}
void makeTree(int me, int head) {
	if (t[me].isChecked)return;
	t[me].isChecked = 1;
	if (t[me].head == -1)t[me].head = head;
	for (int i = 0; i < t[me].member.size(); i++) {
		makeTree(t[me].member[i], t[me].me);
	}
}
/*
그래프 이론
그래프 탐색
트리
너비 우선 탐색
깊이 우선 탐색
*/