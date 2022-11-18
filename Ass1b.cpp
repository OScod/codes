#include<iostream>
#include<bits/stdc++.h>
using namespace std;

// Max function for comparing two numbers
int max(int a,int b)
{
    return (a>b)?a:b;
}

//0/1 Knapsack function
void knapsack(int p[],int w[],int n,int k_capacity)
{
    int i,j,total_profit;
    int a[n+1][k_capacity+1];     //2-d array for matrix implementation
    for(i=0;i<=n;i++)             //Loop for traversing row
    {
        for(j=0;j<=k_capacity;j++)//Loop for traversing columns
        {
            if(i==0 || j==0)
            {
                a[i][j]=0;        //Initialising first row and first column with zero
            }
            else if(w[i-1]<=j)    //If weight is greater than w[i-1] then use formula
            {
                a[i][j]=max(a[i-1][j],(a[i-1][j-w[i-1]]+p[i-1]));
            }
            else                  //Else copy the above value as it is
            {
                a[i][j]=a[i-1][j];
            }
        }
    }
    //The last box of matrix holds the total profit
    //a[n][k_capacity]=Total Profit
    int profit=a[n][k_capacity];
    cout<<"Total profit: "<<profit<<endl;

    cout<<"Matrix generated for Dynamic Programming: "<<endl;
    for(i=0;i<=n;i++)
    {
    	for(j=0;j<=k_capacity;j++)
    	{
    		cout<<a[i][j]<<"\t";
		}
        cout<<endl;
	} 
	cout<<endl;

    //For finding which item is included:
// either the result comes from the top
    // (a[i-1][j]) or from (p[i-1] + a[i-1]
    // [j-w[i-1]]) as in Knapsack table
    // If it comes from the latter one, it means
    // the item is included.
    for(i=n;i>0 && profit>0;i--)
    {
        if(profit==a[i-1][j])
        {
            cout<<"This item is not included "<<i<<" ->0"<<endl;
        }
        else
        {
            //This item is included
            cout<<"This item is included"<<i<<" ->1"<<endl;
            // Since this weight is included its
            // value is deducted
            profit=profit-p[i-1];
            w=w-w[i-1];
        }
    }
}


int main()
{
    int n,k_capacity;
    cout<<"Enter the number of objects: "<<endl;
    cin>>n;
    cout<<"Enter the capacity: "<<endl;
    cin>>k_capacity;
    
    int w[n];    //Weight array
    int p[n];    //Profit array
    
    cout<<"Enter the weights: "<<endl;
    for(int i=0;i<n;i++)
    {
        cin>>w[i];    //Accepting weights values
    }
    
    cout<<"Enter the profit: "<<endl;
    for(int i=0;i<n;i++)
    {
        cin>>p[i];    //Accepting profits values
    }
    
    knapsack(p,w,n,k_capacity);   //Function call for knapsack
    return 0;
}
