#include <iostream>
using namespace std;
#define ll long long
int n;
struct heap_ {
	ll arr[100001];
	ll size;
}typedef H;
void swap(ll *a, ll *b) {
	ll temp = *a;
	*a = *b;
	*b = temp;
}
void insert(H* h, ll data) {
	h->size++;
	ll point = h->size;

	while ((point != 1) && (data < h->arr[point / 2])) {
		h->arr[point] = h->arr[point / 2];
		point /= 2;
	}
	h->arr[point] = data;
}

ll del(H *h) {
	if (h->size == 0) return 0;
	ll ret = h->arr[1];
	h->arr[1] = h->arr[h->size--];
	ll parent = 1;
	ll child;

	while (1) {
		child = parent * 2;
		if (child + 1 <= h->size && h->arr[child] > h->arr[child + 1])
			child++;

		if (child > h->size || h->arr[child] > h->arr[parent]) break;

		swap(&h->arr[parent], &h->arr[child]);
		parent = child;
	}

	return ret;

}
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	cin >> n;
	H h;
	h.size = 0;
	for (int i = 0; i < n; i++) {
		ll temp;
		cin >> temp;
		if (temp != 0) {
			insert(&h, temp);
		}
		else if (temp == 0) {
			cout << del(&h) << "\n";
		}

	}
}

/*
자료 구조
우선순위 큐
*/