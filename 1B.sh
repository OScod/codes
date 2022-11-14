

ch=1
echo enter name of file
read fname
while [ $ch -ne 6 ]; do
echo menu
echo 1.create
echo 2.view
echo 3.search
echo 4.delete
echo 5.modify
echo 6.exit

echo enter your choice
read ch

case $ch in
1)echo enter name
read name
echo enter rollno
read rollno
echo $name $rollno >> $fname
;;
2)cat $fname
;;
3)echo enter the rollno to be searched
read roll
if grep $roll $fname
then
echo Record found
cat $fname
else
echo Record not found
fi
;;
4)echo enter the rollno to be deleted
read roll
if grep $roll $fname
then
grep -v $roll $fname >> temp
rm $fname
mv temp $fname
echo Record deleted
fi
;;
5)echo enter the rollno to be modified
read roll
if grep $roll $fname
then
grep -v $roll $fname >> temp
rm $fname
mv temp $fname
echo enter the new name
read nname
echo enter the new rollno
read nroll
echo $nname $nroll >> $fname
fi
;;
esac
read
done
