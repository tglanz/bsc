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
        return indexOf(ch, 0);
    }

    /**
     * Find the index of the first occurrence of a given character after a given index to start lookin from inclusive
     * @param ch The character to look for in this list
     * @param fromIndex The first index to look from, inclusive
     * @return int The index of the first occurrence of the given character after the given index to start lookin from. -1 if not found
     */
    public int indexOf(int ch, int fromIndex){
        if (_head != null){
            CharNode node = _head;
            int passed = 0;       
            
            // Iterate through all nodes
            while (node != null){
                if (node.getData() == ch){
                    // Check if we are in the allowed range depending on the first index the look from
                    if (fromIndex < passed + node.getValue()){
                        // We have found a node containg the character ch, return its index
                        return passed;
                    }
                }

                // Haven't found ch yet, increment the how much have we passed, and set the next node in the list
                passed += node.getValue();
                node = node.getNext();
            }
        }

        // Head is null, or we just havn't found the character from the first index given...s
        return -1;
    }

    /**
     * Checks wether this list deeply equals to another list
     * @param str The other list to check for equality with
     * @return boolean true if the lists are deeply equal. empty lists are considered equal
     */
    public boolean equals(StringList str){

        // Note: could have used compareTo, but recursion required

        boolean headsAreEqual = false;
        boolean restOfTheListsAreEqual = false;

        if (_head == null && str._head == null){
            // Both lists are empty
            // We assume that two empty lists are considered equal
            headsAreEqual = true;
            restOfTheListsAreEqual = true;
        } else if (_head != null && str._head != null){
            // Check if the heads are equal
            headsAreEqual =
                _head.getData() == str._head.getData() &&
                _head.getValue() == str._head.getValue();

            if (headsAreEqual){
                // Check if the rest of lists are equal
                
                // The recursion step, rely on the correctness of StringList.equals
                restOfTheListsAreEqual = 
                    new StringList(_head.getNext()).equals(new StringList(str._head.getNext()));
            }       
        } // Else: one is empty and the other is not, not equal

        // Code structured this way just to be a bit verbose
        // The lists are equal iff heads are equal, and the lists without the heads are equal
        return headsAreEqual && restOfTheListsAreEqual;
    }

    /**
     * Compares this list to another one.
     * @param str The other list to compare this to
     * @return int comparison result. 0=equal, negative when this list is lt the other,  positive if this list is gt the other
     */
    public int compareTo(StringList str){
        
        // The algorithm basically compares each character at each index
        // by "remembering" how many characters in a node, and compare it's data for node.getValue() times

        CharNode thisNode = _head;
        CharNode otherNode = str._head;

        int thisCharsLeft = -1;
        if (thisNode != null){
            // How many character we haven't compared yet in thisNode
            thisCharsLeft = thisNode.getValue();
        }
        
        int otherCharsLeft = -1;
        if (otherNode != null){
            // How many character we haven't compared yet in otherNode
            otherCharsLeft = otherNode.getValue();
        }

        // As long as there are characters to compare
        while (thisCharsLeft > 0 && otherCharsLeft > 0){
            // Make the comparison
            if (thisNode.getData() != otherNode.getData()){
                // If the character aren't equal return the ordinal comparison
                return thisNode.getData() - otherNode.getData();
            }

            // -1 character left
            --thisCharsLeft;
            --otherCharsLeft;

            // If no more characters in thisNode
            if (thisCharsLeft == 0){
                // Attempt to proceed in the list
                thisNode = thisNode.getNext();
                if (thisNode != null){
                    // If there is still a node to proceed to, there are more characters to check
                    thisCharsLeft += thisNode.getValue();
                }
            }

            // If no more characters in thisNode
            if (otherCharsLeft == 0){
                // Attempt to proceed in the list
                otherNode = otherNode.getNext();
                if (otherNode != null){
                    // If there is still a node to proceed to, there are more characters to check
                    otherCharsLeft += otherNode.getValue();
                }
            }
        }

        // We reach here only if at least one of the lists comparisons have been exhausted, it is the smaller one,
        // unless both exhausted, and they are equal. It is basically depends on the characters left to compare.
        return thisCharsLeft - otherCharsLeft;
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