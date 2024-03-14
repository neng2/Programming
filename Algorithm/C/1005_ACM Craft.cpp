#include <iostream>
#include <algorithm>
#include <cstring>
#include <queue>
using namespace std;

#define N 1001
#define K 1000001
#define MAX(a,b) ((a>b) ? a:b)
typedef struct temp{
    int is_visit=0; //중복해서 거쳐갔을때 큐에 안 쌓이게
    int build_time;
    int child[1000];
    int parent[1000];
    int visit_cnt;
    int child_num,parent_num;
}building;


building n[1001]={0,};
queue<building> sorted;
void track(int target){
    int sum=n[target].build_time;
    while(n[target].visit_cnt!=0){
        n[target].visit_cnt--;
        int temp = n[target].parent[n[target].visit_cnt];//parent 번호 정보
        track(temp); //parent가 없을때까지 tracking
        sum=MAX(sum,n[target].build_time+n[temp].build_time); // parent들의 buildtime과 target의 buildtime 합쳤을때 최대가 되는 값 저장
    }
    if(n[target].visit_cnt==0&&!n[target].is_visit){
        n[target].is_visit=1; //큐 중복 삽입 방지, 최종 방문 시 1로 세팅해서 중복되는 parent를 가지는 child가 방문했을 때 들어오지 않게
        n[target].build_time=sum; //target까지 짓는데 드는 비용을 경로에다가 최대값만 저장
        sorted.push(n[target]);
    }
    
}

int main(){
    int T; cin>>T;
    int struct_num, rule_num;
    for(int i=1;i<=T;i++){
        cin>>struct_num>>rule_num;
        for(int j=1;j<=struct_num;j++){
            cin>>n[j].build_time;
        }
        for(int j=1;j<=rule_num;j++){
            int parent,child;
            cin>>parent>>child;
            n[parent].child[n[parent].child_num++]=child; 
            n[child].visit_cnt++; 
            n[child].parent[n[child].parent_num++]=parent;//parent와 child에 각각 정보 업데이트 (연결된 parent child num, # of parent and child, 방문해야하는 횟수)
        }
        int target; cin>>target;
        track(target);
        cout<<sorted.back().build_time<<endl;
        memset(n,0x00,sizeof(n)); 
        sorted=queue<building>(); // 다음 테스트케이스 위해 큐와 구조체 초기화
    }
}



/*
다이나믹 프로그래밍
그래프 이론
위상 정렬
방향 비순환 그래프
*/

/*
2
8 8
10 20 1 5 8 7 1 43
1 2
1 3
2 4
2 5
3 6
5 7
6 7
7 8
7
4 4
10 1 100 10
1 2
1 3
2 4
3 4
4

///////
120
39
//////////////////////////////

5
3 2
1 2 3
3 2
2 1
1


4 3
5 5 5 5
1 2
1 3
2 3
4

5 10
100000 99999 99997 99994 99990
4 5
3 5
3 4
2 5
2 4
2 3
1 5
1 4
1 3
1 2
4

4 3
1 1 1 1
1 2
3 2
1 4
4
7 8
0 0 0 0 0 0 0
1 2
1 3
2 4
3 4
4 5
4 6
5 7
6 7
7
//////
6
5
399990
2
0


위의 예시를 보자.

이번 게임에서는 다음과 같이 건설 순서 규칙이 주어졌다. 1번 건물의 건설이 완료된다면 2번과 3번의 건설을 시작할수 있다. (동시에 진행이 가능하다) 그리고 4번 건물을 짓기 위해서는 2번과 3번 건물이 모두 건설 완료되어야지만 4번건물의 건설을 시작할수 있다.

따라서 4번건물의 건설을 완료하기 위해서는 우선 처음 1번 건물을 건설하는데 10초가 소요된다. 그리고 2번 건물과 3번 건물을 동시에 건설하기 시작하면 2번은 1초뒤에 건설이 완료되지만 아직 3번 건물이 완료되지 않았으므로 4번 건물을 건설할 수 없다. 3번 건물이 완성되고 나면 그때 4번 건물을 지을수 있으므로 4번 건물이 완성되기까지는 총 120초가 소요된다.

프로게이머 최백준은 애인과의 데이트 비용을 마련하기 위해 서강대학교배 ACM크래프트 대회에 참가했다! 최백준은 화려한 컨트롤 실력을 가지고 있기 때문에 모든 경기에서 특정 건물만 짓는다면 무조건 게임에서 이길 수 있다. 그러나 매 게임마다 특정건물을 짓기 위한 순서가 달라지므로 최백준은 좌절하고 있었다. 백준이를 위해 특정건물을 가장 빨리 지을 때까지 걸리는 최소시간을 알아내는 프로그램을 작성해주자.

///////

첫째 줄에는 테스트케이스의 개수 T가 주어진다. 각 테스트 케이스는 다음과 같이 주어진다. 첫째 줄에 건물의 개수 N과 건물간의 건설순서 규칙의 총 개수 K이 주어진다. (건물의 번호는 1번부터 N번까지 존재한다) 

둘째 줄에는 각 건물당 건설에 걸리는 시간 D1, D2, ..., DN이 공백을 사이로 주어진다. 셋째 줄부터 K+2줄까지 건설순서 X Y가 주어진다. (이는 건물 X를 지은 다음에 건물 Y를 짓는 것이 가능하다는 의미이다) 

마지막 줄에는 백준이가 승리하기 위해 건설해야 할 건물의 번호 W가 주어진다.

*/