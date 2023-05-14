#ifndef TYPES_H
#define TYPES_H

//Node Declaration
struct Node {
  int data;
  struct Node *link;
};

typedef struct Node* NodePtr;

void head_insert(NodePtr& head, int the_number);
//Precondition: The pointer variable head points to
//the head of a linked list.
//Postcondition: A new node containing the_number
//has been added at the head of the linked list.

void insert(NodePtr after_me, int the_number);
//Precondition: after_me points to a node in a linked
//list.
//Postcondition: A new node containing the_number
//has been added after the node pointed to by after_me.

NodePtr search(NodePtr head, int target);
//Precondition: The pointer head points to the head of
//a linked list. The pointer variable in the last node
//is NULL. If the list is empty, then head is NULL.
//Postcondition: Returns a pointer that points to the first
//node that contains the target. If no node contains the
//target, the function returns NULL.

void print(NodePtr head);
//Precondition: The pointer head points to the head of
//a linked list. The pointer variable in the last node
//is NULL. If the list is empty, then head is NULL.
//Postcondition: Print out the list

#endif
