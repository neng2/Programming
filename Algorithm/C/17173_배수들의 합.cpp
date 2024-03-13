#import<ios>
int n,m,x,s,a[1010],i;main(){scanf("%d%d",&n,&m);while(m--){scanf("%d",&x);for(i=x;i<=n;i+=x)if(!a[i]){a[i]=1;s+=i;}}printf("%d",s);}
/*
수학
구현
브루트포스 알고리즘
숏코딩
*/