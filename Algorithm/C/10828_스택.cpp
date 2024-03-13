#include <iostream>
#include <string>
#include <stdlib.h>
using namespace std;

typedef struct NODE {
	int data; // 스택 데이터 저장 공간
	NODE *ptr_prev; // 다음 스택이랑 연결할 포인터
}NODE;

typedef struct STACK {
	int len; // 스택 크기 저장할 변수
	NODE *top; // 가장 위에 있는 스택과 연결할 포인터
}STACK;


int main() {
	int n;
	STACK *a = NULL; // 스택 선언
	a = (STACK*)malloc(sizeof(STACK)); // 스택 동적 할당
	a->len = 0; 
	a->top = NULL;
	string command;

	cin >> n;
	cin.ignore(); // 버퍼 비우기
	for (int i = 0; i < n; i++) {
		getline(cin, command);
		if (command="push") {
            int datanumber;
            cin>>datanumber;
	        cin.ignore(); // 버퍼 비우기
			NODE *newNode = NULL;
			newNode = (NODE*)malloc(sizeof(NODE));
			newNode->data = datanumber; 
			newNode->ptr_prev = a->top;
			a->top = newNode;
			a->len++;
		}
		else if (command == "pop") {
			if (a->top == NULL) {
				cout << -1 << endl;
			}
			else {
				cout << a->top->data << endl; 
				a->top = a->top->ptr_prev; // 가장 위의 스택을 제거하고 밑의 스택과 연결
				a->len--;
			}
		}
		else if (command == "size") {
			cout << a->len << endl;
		}
		else if (command == "empty") {
			if (a->top == NULL) {
				cout << 1 << endl;
			}
			else cout << 0 << endl;
		}
		else if (command == "top") {
			if (a->top == NULL) {
				cout << -1 << endl;
			}
			else cout << a->top->data << endl;
		}
	}
}

/*
구현
자료 구조
스택
*/