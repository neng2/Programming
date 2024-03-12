fn main() {
    let s = String::from("hello");

    println!("{},{}",str_len(s.clone()),s);
}
fn str_len(s:String)->usize{
    s.len()
}
fn straddr_len(s:&String)->usize{
    s.len()
}