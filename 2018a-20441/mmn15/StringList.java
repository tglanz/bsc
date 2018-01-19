/**
 * A linked list of CharNode. Represents a string.
 */
public class StringList {

    private CharNode _head;

    /**
     * Creates a new, empty, StringList
     */
    public StringList()
    {
        _head = null;
    }

    /**
     * Creates a new StringList, starting with the given node
     * @param node The head of the list
     */
    public StringList(CharNode node) {
        if (node == null){
            _head = null;
        }
        else {
            _head = new CharNode(node.getData(), node.getValue(), null);

            for (CharNode ptr = node.getNext(), last = _head; ptr != null; ptr = ptr.getNext()) {
                last.setNext(new CharNode(ptr.getData(), ptr.getValue(), ptr.getNext()));
                last = last.getNext();
            }
        }
    }

    /**
     * Constructrs a new instance of StringList, representing a given string
     * @param s The string this list based on
     */
    public StringList(String s){
        int length = s.length();

        if (length == 0){
            _head = null;
            return;
        }

        _head = new CharNode(s.charAt(0), 1, null);
        CharNode last = _head;

        for (int idx = 1; idx < length; ++idx){
            char current = s.charAt(idx);

            if (current == last.getData()){
                last.setValue(last.getValue() + 1);
            }
            else {
                last.setNext(new CharNode(current, 1, null));
                last = last.getNext();
            }
        }
    }

    /**
     * Deep copies another given StringList
     * @param other The StringList to copy
     */
    public StringList (StringList other){
        if (other._head == null){
            _head = null;
            return;
        }

        _head = new CharNode(other._head.getData(), other._head.getValue(), null);

        CharNode otherNode = other._head.getNext();
        CharNode thisLastNode = _head;

        while (otherNode != null){
            CharNode newNode = new CharNode(otherNode.getData(), otherNode.getValue(), null);
            thisLastNode.setNext(newNode);
            thisLastNode = newNode;
            otherNode = otherNode.getNext();
        }
    }

    /**
     * Gets the char at the given index
     * @param i The index to get the char at
     * @return char The character at the given index
     */
    public char charAt(int i){
        if (_head == null){
            return '\0';
        }

        CharNode node = _head;
        int passed = _head.getValue();

        // We strongly rely on the assumption that i is in the range of the list.
        // If it isn't we will get null references, or in case it is negative, the first element always
        while (i >= passed){
            node = node.getNext();
            passed += node.getValue();
        }

        // The first node that doesn't meet the above condition is the node containing
        // the character at the index i
        return node.getData();
    }

    /**
     * Concatenate this with another given list, and creates a new StringList object
     * @param str The list to concatenate this with
     * @return StringList a new StringList object
     */
    public StringList concat (StringList str){

        // If this is an empty list, concatenating str to it is just str
        if (_head == null){
            return str;
        }

        // Deep copy, don't change this
        StringList retVal = new StringList(this);

        // If it isn't Empty, traverse the return value to find the last value
        if (str._head != null){
            // Traverse the list to get to the last node
            CharNode node = retVal._head;
            while (node.getNext() != null){
                node = node.getNext();
            }

            node.setNext(str._head);
        } // else: str is empty, we have nothing to concatenate with, leave retVal as is

        return retVal;
    }

    /**
     * Find the index of the first occurrence of a given character
     * @param ch The character to look for in this list
     * @return int The index of the first occurrence of the given character. -1 if not found
     */
    public int indexOf(int ch){
        if (_head != null){
            CharNode node = _head;
            int passed = 0;       
            
            // Iterate through all nodes
            while (node != null){
                if (node.getData() == ch){
                    // We have found a node containg the character ch, return its index
                    return passed;
                }

                // Haven't found ch yet, increment the how much have we passed, and set the next node in the list
                passed += node.getValue();
                node = node.getNext();
            }
        }

        // Head is null or we just havn't found the character.
        // Anyway, character is not in this list..
        return -1;
    }

    public void DEBUG_PrintStuff(){
        if (_head == null){
            System.out.println("EMPTY");
            return;
        }

        CharNode next = _head;
        while (next != null){
            next.Debug();
            next = next.getNext();
        }
    }

    public void DEBUG_SetNext(CharNode next){
        if (_head == null){
            _head = next;
            return;
        }

        CharNode node = _head;
        while (node.getNext() != null){
            node = node.getNext();
        }

        node.setNext(next);
    }
}