#include "types.h"
#include <iostream>
using namespace std;

//Function Definition
void head_insert(NodePtr& head, int the_number) {
  NodePtr temp_ptr;
  temp_ptr = new Node;

  temp_ptr->data = the_number;

  temp_ptr->link = head;
  head = temp_ptr;
}

void insert(NodePtr after_me, int the_number) {
  if (after_me == NULL) return;
  NodePtr temp_ptr;
  temp_ptr = new Node;

  temp_ptr->data = the_number;

  temp_ptr->link = after_me->link;
  after_me->link = temp_ptr;
}

NodePtr search(NodePtr head, int target) {
  NodePtr here = head;

  while (here->data != target &&
	 here->link != NULL)
    here = here->link;

  if (here->data == target)
    return here;
  else
    return NULL;
}

void print(NodePtr head) {
  NodePtr here = head;

  if (here == NULL) { //special case: empty list
    cout << "it's an empty list" << endl;
  }
  else {
    while (here->link != NULL) {
      cout << here->data << ",";
      here = here->link;
    }

    //print out the last item
    cout << here->data << endl;
  }
}


int main(){
  //create a linked list
  NodePtr head = NULL;

  //keep inserting nodes from head
  for(int i = 0; i < 10; i++)
    head_insert(head, i);
  print(head);
  
  //keep inserting nodes from head
  for(int i = 0; i <= 10; i++)
    insert(search(head, i), -i);
  print(head);
  
  return 0;
}
