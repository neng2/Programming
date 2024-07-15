#include <iostream>
#include <algorithm>

using namespace std;

int min_ = 999999999;
int minper;
int N,M;
int rel[101][101];

int main(){
	cin>>N>>M;
	for(int i = 1; i <= M; i++){
		int point, target;
		cin >> point >> target;
		rel[point][target] = 1;
		rel[target][point] = 1;
	}
	for(int i = 1; i <= N; i++){
		for(int j = 1; j <= N; j++){
			for(int k = 1; k <= N; k++){
			    if(k == j) continue;
				else if(rel[j][i]&&rel[i][k]){
					if(!rel[j][k]) rel[j][k] = rel[j][i]+rel[i][k];
					else if(rel[j][k] >  rel[j][i]+rel[i][k]){
                        rel[j][k] = rel[j][i]+rel[i][k];
					}
		
                }
            }
		}
	}
	for(int i = 1; i <= N; i++){
		int sum = 0;
		for(int j = 1; j <= N; j++){
			sum += rel[i][j];
		}
		if(min_ > sum){
			min_ = sum;
			minper = i;
		}
	}
	cout << minper << endl;
}

/*
그래프 이론
그래프 탐색
너비 우선 탐색
최단 경로
플로이드–워셜
*/