#include <iostream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;


typedef struct tree {
	char data;
	tree *left;
	tree *right;
}tree;
tree initTree(char root);
void createTree(tree *a, char left, char right);
void preOrder(tree *p);
void inOrder(tree *p);
void postOrder(tree *p);
int n, result;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	tree *t = new tree[26];
	t->data = 'A';
	t->left = NULL;
	t->right = NULL;
	cin >> n;
	cin.ignore();
	for (int i = 0; i < n; i++) {
		string input;
		char tok;
		vector<char> word;
		getline(cin, input);
		for (stringstream l(input); (l >> tok);) {
			word.push_back(tok);
		}
		createTree(t+(word.at(0)-'A'), word.at(1), word.at(2));
	}
	preOrder(t);
	cout << "\n";
	inOrder(t);
	cout << "\n";
	postOrder(t);
	cout << "\n";
	//printf()
	delete t;
}
void preOrder(tree *p) {
	//printf("%c", p->data);
	cout << p->data;
	if (p->left != NULL)preOrder(p->left);
	if (p->right != NULL)preOrder(p->right);
	return;
}
void inOrder(tree *p) {
	if (p->left != NULL)inOrder(p->left);

	cout << p->data;
	if (p->right != NULL)inOrder(p->right);
	return;
}
void postOrder(tree *p) {
	if (p->left != NULL)postOrder(p->left);
	if (p->right != NULL)postOrder(p->right);
	cout << p->data;
	return;
}
tree initTree(char root) {
	tree dumy;
	dumy.data = root;
	dumy.left = NULL;
	dumy.right = NULL;
	return dumy;
}
void createTree(tree *a, char left, char right) {

	if (left != '.') {
		*(a + left - (a->data)) = initTree(left);
		a->left = a + left - (a->data);
	}
	else a->left = NULL;

	if (right != '.') {
		*(a + right - (a->data)) = initTree(right);
		a->right = a + right - (a->data);
	}
	else a->right = NULL;
}


/*
트리
재귀
*/