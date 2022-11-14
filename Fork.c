//Program for sorting elements of any array using fork ()
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
int total;
int partition(int numbers[],int left,int right)
{
int pivot,Temp,pivotP=left;
pivot = numbers[left++];
printf("\nPivot (%3d)=> ",pivot);
while (left <= right)
{
while ((numbers[right] <= pivot) && (left <= right))
right--;
while ((numbers[left] > pivot) && (left <= right))
left++;
if(left<right)
{
Temp=numbers[right];
numbers[right] = numbers[left];
numbers[left]=Temp;
left++;right--;
}
}
if(pivotP!=right)
{
Temp=numbers[right];
numbers[right] = numbers[pivotP];
numbers[pivotP]=Temp;
}
pivot = right;
return pivot;
}
// for sorting Quick Sort Algorithm is used
void quicksort(int numbers[], int left, int right)
{
int i,pivot;
static int Pass=1;
pivot=partition(numbers,left,right);
printf(" Pass %2d: ",Pass++);
for(i=0;i<total;i++)
printf("%3d ",numbers[i]);
if (left < pivot)
quicksort(numbers, left, pivot-1);
if (right > pivot)
quicksort(numbers, pivot+1, right);
}
int merge_sort(int arr[],int low,int high)
{
int mid;
if(low<high) {
mid=(low+high)/2;
// Divide and Conquer
merge_sort(arr,low,mid);
merge_sort(arr,mid+1,high);
// Combine
merge(arr,low,mid,high);
}
return 0;
}
int merge(int arr[],int l,int m,int h)
{
int arr1[10],arr2[10]; // Two temporary arrays to hold the two arrays to be merged
int n1,n2,i,j,k;
n1=m-l+1;
n2=h-m;
for(i=0; i<n1; i++)
arr1[i]=arr[l+i];
for(j=0; j<n2; j++)
arr2[j]=arr[m+j+1];
arr1[i]=9999; // To mark the end of each temporary array
arr2[j]=9999;
i=0;
j=0;
for(k=l; k<=h; k++) { //process of combining two sorted arrays
if(arr1[i]<=arr2[j])
arr[k]=arr1[i++];
else
arr[k]=arr2[j++];
}
return 0;
}
int main()
{
pid_t pid;
int status;
int a[10],i;
printf("\nEnter Total Number of Element:");
scanf("%d",&total);
printf("\nEnter %d Element of Array",total);
for(i=0;i<total;i++)
scanf("%d",&a[i]);
pid=fork();
if(pid==-1)
{
printf("\nError");
}
else if(pid>0)
{
//sleep(2);
printf("\nParent Process");
printf("\nChild Process Id: %d Parent Process Id:%d",getpid(),getppid());
quicksort(a,0,total-1);
pid=wait(&status);
}
else
{
//sleep(5);
printf("\nChild Process");
printf("\nChild Process Id: %d Parent Process Id:%d",getpid(),getppid());
merge_sort(a,0,total-1); // sort the array
printf("\nSorted array:"); // print sorted array
for(i=0; i<total; i++)
printf("%d\t",a[i]);
}
return 0;
}

