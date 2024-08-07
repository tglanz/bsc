/**
 * Represents a singe character's linked list node
 */
public class CharNode {

    private char _data;
    private int _value;
    private CharNode _next;

    /**
     * A constructor for CharNode
     *
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @param c The character contained in the node
     * @param val The number of squential appearances
     * @param n The next node in the list
     */
    public CharNode(char c, int val, CharNode n) {
        _data = c;
        _value = val;
        _next = n;
    }

    /**
     * Gets the next node in the list
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @return CharNode The next node in the list
     */
    public CharNode getNext() {
        return _next;
    }

    /**
     * Sets the next node in the list
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @param node The next node to set in the list
     */
    public void setNext(CharNode node) {
        _next = node;
    }

    /**
     * Gets the number of sequential appearances of the character in the list
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @return int The number of sequential appearances of the character in the list
     */
    public int getValue() {
        return _value;
    }

    /**
     * Sets the number of sequential appearances of the character in the list
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @param v the number of sequential appearances of the character in the list
     */
    public void setValue(int v) {
        _value = v;
    }

    /**
     * Gets the character contained in the node
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @return char the character contained in the node
     */
    public char getData() {
        return _data;
    }

    /**
     * Sets the character contained in the node
     * 
     * Time complexity     : O(1)
     * Memory complexity   : O(1)
     * 
     * @param c the character contained in the node
     */
    public void setData(char c) {
        _data = c;
    }
}
