/*=============================================================================
Assignment   : maman 11 - the practical part 

Author       : Tal Glanzman
Id           : 302800354 

Course       : Operating Systems - 20594 

Description  : Implement a thread switch mechanism.
               We assume the program to have two globals: current_thread and next_thread
               of struct thread (defined in uthread.c).            
               Practically, we will push the state onto the current stack,
               replace next_thread to be current_thread, and pop it's state from the stack.
============================================================================*/
    .text

    .globl thread_switch
thread_switch:
     
    /* push all registers that are in the current thread's stack frame */
    pushal

    /* the reason for this is to keep in memory where the current
     * stack pointer is at, which will be loaded again on a later
     * phase (practically, when current will be the next next)
     * c-equiv: current_thread->sp = esp;
     */
    movl current_thread, %eax
    movl %esp, (%eax)

    /* replace the current thread to be the next thread
     * c-equiv: current_thread = next_thread;
     */
    movl next_thread, %eax
    movl %eax, current_thread

    /* load the stack pointer that is in the thread's struct
     * c-equiv: esp = current_thread->sp
     */
    movl current_thread, %eax
    movl (%eax), %esp

    /* pop all registers that are in the thread's stack frame */
    popal

    /* set next_thread to 0 */
    movl next_thread, %eax
    movl 0, %eax

    /* pop return address from stack */
    ret

